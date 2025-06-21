import sys
from pathlib import Path
import os
import datetime
import io

sys.path.append(Path(__file__).resolve().parents[0].joinpath("src").as_posix())
import logging
from shutil import rmtree
from argparse import ArgumentParser
from pprint import pformat
import yaml
import copy
from itertools import product, combinations
import math

import torch
from torch.utils.data import DataLoader
import tqdm
import torchvision as tv
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from blackbox_locking.models import (
    get_resnet18,
    get_resnet50,
    get_vgg16,
    get_efficientnet_b0,
    get_densenet121,
    get_mobilenet_v2,
    get_mobilenet_v3_small,
    get_mobilenet_v3_large,
)
from blackbox_locking.datasets import get_cifar10, get_cifar100, get_imagenet
from blackbox_locking.utils import set_seed
from blackbox_locking.logging import set_logging_verbosity
from blackbox_locking.loss_fn import one_vs_rest_cross_entropy_loss, one_vs_one_logits_difference_gain
from blackbox_locking.quantize import quantize_model
from blackbox_locking.identifier import DeviceIdentifierOneVsOne
#os.environ["NCCL_PORT_RANGE"] = "29500-30500"

print("NCCL_PORT_RANGE:", os.environ.get("NCCL_PORT_RANGE"))
print("NCCL_SOCKET_IFNAME:", os.environ.get("NCCL_SOCKET_IFNAME"))
logger = logging.getLogger("blackbox_locking." + __name__)

TASK_TO_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet": None,
}
DEFAULT_SAVE_DIR = "results/border_images_batchsize"
DEFAULT_CKPT ="checkpoints/default/resnet18-cifar10.pt"
OTHER_CKPT = "checkpoints/default/resnet18-cifar10.pt"

def setup_distributed(rank, world_size, master_addr, master_port):
    init_uri = f"tcp://{master_addr}:{master_port}"
    print(f"Initializing process group on rank {rank} via {init_uri}…")
    dist.init_process_group(
        backend="nccl",
        init_method=init_uri,
        rank=rank,
        world_size=world_size,
    )
    print(f"Process group initialized on rank {rank}")
    
def cleanup_distributed():
    dist.destroy_process_group()

def run_quantization_and_testing(rank, world_size, model, q_tags, test_dataloader, devices):

    device = devices[rank]
    torch.cuda.set_device(device)

    test_acc = {}
    criterion = torch.nn.functional.cross_entropy

    for q_tag in q_tags:
        q_model = copy.deepcopy(model)
        quantize_model(q_model, q_tag, layers_to_ignore=[])
        q_model = q_model.to(device)

        _, acc = validate(q_model, test_dataloader, criterion, device, description=f"Test ({q_tag})", silent=True)

        q_model = q_model.cpu()
        test_acc[q_tag] = acc

    if rank == 0:
        logger.info(f"[Rank 0] Test accuracy of quantized models:\n{pformat(test_acc, sort_dicts=False)}")
    elif rank == 1:
        logger.info(f"[Rank 1] Test accuracy of quantized models:\n{pformat(test_acc, sort_dicts=False)}")

def train_one_epoch(model, train_loader, optimizer, lr_scheduler, criterion, num_epochs, device):
    model = model.to(device)
    model.train()
    running_loss = 0.0
    prog_bar = tqdm.tqdm(train_loader, desc="Training", leave=True)
    for images, labels in prog_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        prog_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    lr_scheduler.step()
    avg_loss = running_loss / len(train_loader)
    logger.info(f"Epoch {num_epochs + 1} training loss: {avg_loss}")
    return avg_loss


@torch.no_grad
def validate(model, val_loader, criterion, device, description="Validation", silent=False):
    model.eval()
    model = model.to(device)
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm.tqdm(val_loader, desc=description, leave=True):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.cpu().item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().cpu().item()
    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    if not silent:
        logger.info(f"Validation loss: {avg_loss}, accuracy: {accuracy}")
    return avg_loss, accuracy


def create_model(model_name: str, num_classes: int):
    match model_name:
        case "vgg16":
            model, transform = get_vgg16(num_classes)
        case "resnet18":
            model, transform = get_resnet18(num_classes)
        case "resnet50":
            model, transform = get_resnet50(num_classes)
        case "efficientnet_b0":
            model, transform = get_efficientnet_b0(num_classes)
        case "densenet121":
            model, transform = get_densenet121(num_classes)
        case "mobilenet_v2":
            model, transform = get_mobilenet_v2(num_classes)
        case "mobilenetv3small":
            model, transform = get_mobilenet_v3_small(num_classes)
        case "mobilenetv3large":
            model, transform = get_mobilenet_v3_large(num_classes)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    return model, transform
    


def fine_tune_model(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    save_dir: Path,
    num_epochs: int = 3,
    lr=1e-2,
    device="cuda:0",
) -> dict[str, float]:
    model = model.to(device)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_ckpt = save_dir.joinpath("best_model.pth")

    criterion = torch.nn.functional.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=num_epochs)

    for epoch in range(num_epochs):
        train_one_epoch(model, train_dataloader, optimizer, lr_scheduler, criterion, epoch, device)
        val_loss, val_acc = validate(model, val_dataloader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_ckpt)

    model.load_state_dict(torch.load(best_model_ckpt))
    model = model.to(device)

    test_loss, test_acc = validate(model, test_dataloader, criterion, device, description="Test", silent=True)

    logger.info(f"Best validation loss: {best_val_loss}, accuracy: {best_val_acc}")
    logger.info(f"Test loss: {test_loss}, accuracy: {test_acc}")

    metrics = dict(
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        test_loss=test_loss,
        test_acc=test_acc,
    )

    metric_path = save_dir.joinpath("metrics.yaml")
    with metric_path.open("w") as f:
        yaml.safe_dump(metrics, f)

    return metrics


@torch.no_grad
def quantize_img_and_to_tensor(img: torch.Tensor) -> torch.Tensor:
    assert img.ndim == 4
    # assert img.shape[0] == 1

    img = img.mul(255).round().clamp(0, 255).to(torch.uint8)
    img = img.float().div(255.0)

    return img

@torch.no_grad()
def check_batch_order_consistency(model: torch.nn.Module, batch: torch.Tensor, device: str) -> bool:
    """
    Check if model predictions are consistent across different batch orderings.
    Returns True if predictions are consistent, False otherwise.
    """
    model = model.to(device)
    model.eval()
    
    # Original order
    orig_output = model(batch).to(torch.float64)  # Use double precision
    
    # Random permutation
    idx = torch.randperm(batch.size(0))
    shuffled_batch = batch[idx]
    shuffled_output = model(shuffled_batch).to(torch.float64)
    # Restore original order
    restored_output = shuffled_output[torch.argsort(idx)]
    
    # Compare logits with high precision
    max_diff = torch.max(torch.abs(orig_output - restored_output)).item()
    is_consistent = max_diff < 1e-50  # Adjust threshold as needed
    
    if not is_consistent:
        print(f"Original logits:\n{orig_output}\nRestored logits:\n{restored_output}\nMax difference: {max_diff}")
    
    return is_consistent, max_diff


@torch.no_grad
def check_border_image_batch(
    rank,
    target_model: torch.nn.Module,
    remainder_models: list[torch.nn.Module],
    artificial_img: torch.Tensor,
    label: torch.Tensor,
    eval_batch_size: int,
):
    """
    Return:
        find_any_border_img: bool, whether any border image is found
        is_border_img: torch.Tensor, whether artificial_img is a border image, (B,)
        artificial_img: torch.Tensor, quantized image, (B, C, H, W)
        r_mode_agree: list[bool], whether all remainder models agree on the prediction, (B,)
        r_models_agree_with_gt: list[bool], whether all remainder models agree with the ground truth, (B,)
        t_pred_label: list[int], target model prediction, (B,)
        r_pred_label_list: list[list[int]], remainder model predictions, [(B,), ...]
    """
    assert artificial_img.ndim == 4
    # assert artificial_img.shape[0] == 1
    assert label.ndim == 1
    # assert label.shape[0] == 1
    assert artificial_img.shape[0] == label.shape[0]
    if rank ==0:
        artificial_img = quantize_img_and_to_tensor(artificial_img)
        #if eval_batch_size > 1:
        #    is_consistent = check_batch_order_consistency(target_model, artificial_img, device)
        #    if not is_consistent:
        #        logger.warning("Model predictions are not consistent across batch orderings!")
        dist.send(tensor=artificial_img, dst=1)
        if eval_batch_size == 1:
            # check each image separately
            t_logits_list = []
            for img in artificial_img:
                img = img.unsqueeze(0) 
                logits = target_model(img)
                t_logits_list.append(logits)
            t_logits = torch.cat(t_logits_list, dim=0)
        else:
            t_logits = target_model(artificial_img)
        r_logits_list = torch.zeros(len(remainder_models), t_logits.shape[0],t_logits.shape[1]).to(device)
        dist.recv(tensor=r_logits_list, src=1)
        r_logits_list = torch.unbind(r_logits_list, dim=0)
    elif rank==1:
        recv_img = torch.zeros_like(artificial_img).to(device)
        dist.recv(tensor=recv_img, src=0)
        recv_img = recv_img.to(device)
        if eval_batch_size == 1:
            # this produces different logits from the other code below
            """r_logits_list = []
            for r_model in remainder_models:
                logits_list = []
                for img in recv_img:
                    img = img.unsqueeze(0)
                    logits = target_model(img)
                    logits_list.append(logits)
                r_logits_list.append(torch.cat(logits_list, dim=0))
            r_logits_tensor = torch.stack(r_logits_list, dim=0).to(device)"""
            r_logits_list = []
            for img in recv_img:
                img = img.unsqueeze(0) 
                logits = remainder_models[0](img)
                r_logits_list.append(logits)
        else:
            r_logits_list = [r_model(recv_img) for r_model in remainder_models]  # [(B, num_classes), ...]

        r_logits = torch.cat(r_logits_list, dim=0)
        r_logits_tensor = r_logits.unsqueeze(0)
        dist.send(tensor=r_logits_tensor, dst=0)
        return None

    t_pred_label = torch.argmax(t_logits, dim=1)
    r_pred_label_list = [torch.argmax(r_logits, dim=1) for r_logits in r_logits_list]

    r_models_agree_with_gt = None  # agree with ground truth
    for r_pred_label in r_pred_label_list:
        if r_models_agree_with_gt is None:
            r_models_agree_with_gt = r_pred_label == label  # (B,)
        else:
            r_models_agree_with_gt = r_models_agree_with_gt & (r_pred_label == label)

    r_models_agree = None
    for r_pred_label in r_pred_label_list:
        if r_models_agree is None:
            r_models_agree = r_pred_label == t_pred_label  # (B,)
        else:
            r_models_agree = r_models_agree & (r_pred_label == t_pred_label)

    is_border_img = None
    for r_pred_label in r_pred_label_list:
        if is_border_img is None:
            is_border_img = r_pred_label != t_pred_label
        else:
            is_border_img = is_border_img & (r_pred_label != t_pred_label)

    find_any_border_img = is_border_img.any().item()

    r_pred_label_list = [r_pred_label.cpu().tolist() for r_pred_label in r_pred_label_list]
    num_border_img = is_border_img.sum().item()
    if num_border_img>0:
        border_img_indices = torch.nonzero(is_border_img, as_tuple=False).squeeze().tolist()
        print(f"Border images found at indices: {border_img_indices}")
        #print(artificial_img[border_img_indices])
        print("Logits: ",t_logits[border_img_indices], r_logits_list[0][border_img_indices])
        r_pred_label_tensor = torch.tensor(r_pred_label_list[0])
        print("Labels: ", t_pred_label[border_img_indices], r_pred_label_tensor[border_img_indices])
    #print("Logits: ",t_logits, r_logits_list[0])
    #r_pred_label_tensor = torch.tensor(r_pred_label_list[0])
    #print("Labels: ", t_pred_label, r_pred_label_tensor)       
    return (
        find_any_border_img,
        num_border_img,
        is_border_img.cpu().tolist(),
        artificial_img,
        r_models_agree.cpu().tolist(),
        r_models_agree_with_gt.cpu().tolist(),
        t_pred_label.cpu().tolist(),
        r_pred_label_list,
    )



def engineer_border_image_via_1vs1_pgd_distributed(
    model_1: torch.nn.Module,
    model_2: torch.nn.Module,
    tag_1: str,
    tag_2: str,
    src_img: torch.Tensor,
    label: torch.Tensor,
    num_iters: int,
    start_lr: float,
    end_lr: float,
    prog_bar_desc: str,
    check_every: int,
    rank: int,
    device: str,
    eval_batch_size: int,
):
    
    model_1.eval()
    model_2.eval()
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)
    batch_size = src_img.shape[0]

    artificial_img = src_img.clone().detach().to(device)
    prog_bar = tqdm.tqdm(range(num_iters), desc=prog_bar_desc)

    lr = start_lr

    for iter_idx in prog_bar:
        artificial_img.requires_grad_(True)
        model_1.zero_grad()
        model_2.zero_grad()
        
        if rank == 0:
            logits_1 = model_1(artificial_img)
            dist.send(tensor=artificial_img, dst=1)  # Send image to rank 1
            logits_2 = torch.zeros_like(logits_1).to(device)
            dist.recv(tensor=logits_2, src=1)  # Receive logits_2 from rank 1
            logits_1, logits_2 = logits_1.to(device), logits_2.to(device)
            loss = one_vs_one_logits_difference_gain(logits_1, logits_2)
            loss.backward()
    
            artificial_img = artificial_img + lr * artificial_img.grad.sign()
            artificial_img = artificial_img.clamp(0, 1).detach()
    
            prog_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
            with torch.no_grad():
                if (iter_idx + 1) % check_every == 0 or iter_idx == num_iters - 1:
                    find_any_border_img, num_border_img, is_border_img, quantized_img, r_models_agree, r_models_agree_with_gt, t_pred_label, r_pred_label_list = check_border_image_batch(
                        0,model_1, [model_2], artificial_img, label, eval_batch_size)
    
                    if find_any_border_img:
                        prog_bar.close()
                        # Here I need to signal rank 1 that we are done
                        done_tensor = torch.ones(1, dtype=torch.bool).to(device)
                        print("Send we are done")
                        dist.send(tensor=done_tensor, dst=1)
                        return dict(
                            is_border_image=is_border_img,
                            num_border_images=num_border_img,
                            border_image=quantized_img.cpu(),
                            model_1_tag=f"{tag_1}_g1_{batch_size}",
                            model_2_tag=f"{tag_2}_g2_{batch_size}",
                            model_1_pred_label=t_pred_label,
                            model_2_pred_label=r_pred_label_list[0],
                            ground_truth_label=label.cpu().tolist(),
                            num_iters=iter_idx + 1,
                        )
                    else:
                        done_tensor = torch.zeros(1, dtype=torch.bool).to(device)
                        #print("Send we are not done yet: model 1 pred label, model 2 pred label, ground truth label: ",t_pred_label,r_pred_label_list[0],label.cpu().tolist())
                        dist.send(tensor=done_tensor, dst=1)
                else:
                    done_tensor = torch.zeros(1, dtype=torch.bool).to(device)
                    dist.send(tensor=done_tensor, dst=1)
            lr = calculate_current_linear_lr(start_lr=start_lr, end_lr=end_lr, total_steps=num_iters, current_step=iter_idx)
        else: # rank==1
            recv_img = torch.zeros_like(artificial_img).to(device)
            dist.recv(tensor=recv_img, src=0)  # Receive image from rank 0
            recv_img = recv_img.to(device)
            logits_2 = model_2(recv_img)
            dist.send(tensor=logits_2 , dst=0)  # Send logits_2 to rank 0
            with torch.no_grad():
                if (iter_idx + 1) % check_every == 0 or iter_idx == num_iters - 1:
                    check_border_image_batch(1,model_1, [model_2], artificial_img, label, eval_batch_size)
            done_tensor = torch.zeros(1, dtype=torch.bool).to(device)
            dist.recv(tensor=done_tensor, src=0)
            if done_tensor.item():
                return None
    
    model_1 = model_1.cpu()
    model_1 = model_2.cpu()
    return None

def engineer_border_image_via_1_vs_rest(*args, **kwargs):
    raise NotImplementedError

def main_worker(rank,world_size, model, model_tags, dataloader, noise_scale, start_lr, end_lr, num_iters_per_img, num_tries_per_model, check_every, devices, other_models, other_tags, eval_batch_size):
    # This will be called by each process
    border_img_group = engineer_border_image_group_via_1vs1_pgd_distributed(
        rank=rank,
        world_size=world_size,
        model=model,
        model_tags=model_tags,
        dataloader=dataloader,
        noise_scale=noise_scale,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iters_per_img=num_iters_per_img,
        num_tries_per_model=num_tries_per_model,
        check_every=check_every,
        devices=devices,
        eval_batch_size = eval_batch_size,
        other_models = other_models,
        other_tags = other_tags,
    )

    if rank == 0:
        return border_img_group

def engineer_border_image_group_via_1vs1_pgd_distributed(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    model_tags: list[str],
    dataloader: DataLoader,
    noise_scale: float,
    start_lr: float,
    end_lr: float,
    num_iters_per_img: int,
    num_tries_per_model: int,
    check_every: int,
    devices: list[str],
    eval_batch_size: int,
    other_models: None | list[torch.nn.Module] = None,
    other_tags: None | list[str] = None,
    
):
    print("My rank: ",rank)

    device = devices[rank]
    torch.cuda.set_device(device)
    model = model.cpu()
    model.eval()

    border_img_group = []
    total_num_models = len(model_tags)
    total_num_pairs = total_num_models * total_num_models  # Including self-comparison

    for i, (model_pair_idx_1, model_pair_idx_2) in enumerate(product(range(total_num_models), repeat=2)):
        logger.info(
            f"🚀 Engineering border images for {i+1}/{total_num_pairs} model pair: {model_tags[model_pair_idx_1]} vs {model_tags[model_pair_idx_2]}"
        )

        tag_1, tag_2 = model_tags[model_pair_idx_1], model_tags[model_pair_idx_2]
        model_1 = copy.deepcopy(model)
        model_2 = copy.deepcopy(model)
        model_1.eval()
        model_2.eval()
        quantize_model(model_1, tag_1, layers_to_ignore=[])
        quantize_model(model_2, tag_2, layers_to_ignore=[])
        model_1 = model_1.to(device)
        model_2 = model_2.to(device)

        total_num_iters = 0
        for try_idx, (src_img, label) in enumerate(dataloader):
            # src_img: (B, C, H, W), label: (B,)
            src_img, label = src_img.to(device), label.to(device)
            noise = (torch.rand_like(src_img) - 0.5) * 2 * noise_scale
            src_img = src_img.add(noise).clamp(0, 1)

            border_img = engineer_border_image_via_1vs1_pgd_distributed(
                model_1=model_1,
                model_2=model_2,
                tag_1=tag_1,
                tag_2=tag_2,
                src_img=src_img,
                label=label,
                num_iters=num_iters_per_img,
                start_lr=start_lr,
                end_lr=end_lr,
                prog_bar_desc=f"Engineering border image (try {try_idx+1}/{num_tries_per_model})",
                check_every=check_every,
                rank=rank,
                device=device,
                eval_batch_size=eval_batch_size,
            )

            if border_img is not None: # runs only for rank 0
                done_tensor = torch.ones(1, dtype=torch.bool).to(device)
                print("Send we are done")
                dist.send(tensor=done_tensor, dst=1)
                border_img["num_iters"] = total_num_iters + border_img["num_iters"]
                border_img_group.append(border_img)
                if other_models is None or other_tags is None:
                    border_img_info = {k: v for k, v in border_img.items() if k != "border_image"}
                    logger.info(
                        f"✅ Found border image for {i+1}/{total_num_pairs}-th model pair:\n{pformat(border_img_info, sort_dicts=False, compact=True)}"
                    )
                else:# if there are other_models and we want to test transferability
                    # Probably need to do this separately for rank 0 and rank 1?
                    # But tag_1 basically always belongs to rank 0 and tag_2 to rank 1
                    # So only need to do for one tag for each rank

                    # First send the dict to rank 1 as well
                    print("Check for transferability")
                    send_dict(border_img, dst=1, device=device)
                    other_q_models = []
                    other_q_model_tags = []
                    for other_model in other_models:
                        for other_tag in [tag_1]: # only tag_1!!
                            other_q_model = copy.deepcopy(other_model)
                            quantize_model(other_q_model, other_tag, layers_to_ignore=[])
                            other_q_model.eval()
                            other_q_models.append(other_q_model)
                            other_q_model_tags.append(other_tag)
                    other_q_model = None
                    # this preferably only call for rank 0?
                    find_any_fully_transferable, best_transferable_img = find_best_transferable_border_image(
                        rank,
                        border_img,
                        other_q_models,
                        [tag_1,tag_2],
                        device,
                        eval_batch_size,
                        src_img.shape[0]
                    )
                    border_img["find_any_fully_transferable"] = find_any_fully_transferable
                    border_img["best_transferable_image"] = best_transferable_img

                    del other_q_models
                    del other_q_model_tags

                    border_img_info = {k: v for k, v in border_img.items() if k != "border_image"}
                    logger.info(
                        f"✅ Found border image for {i+1}/{total_num_pairs}-th model pair:\n{pformat(border_img_info, sort_dicts=False, compact=True)}"
                    )
                break
            else:
                if rank==0:
                    done_tensor = torch.zeros(1, dtype=torch.bool).to(device)
                    print("Send we are not done yet")
                    dist.send(tensor=done_tensor, dst=1)
                elif rank==1:
                    done_tensor = torch.zeros(1, dtype=torch.bool).to(device)
                    print("Receive whether we are done or not")
                    dist.recv(tensor=done_tensor, src=0)
                    if done_tensor.item():
                        if other_models is not None:
                            # First we need to receive the border image dict
                            border_img = recv_dict(src=0, device=device)
                            other_q_models = []
                            other_q_model_tags = []
                            for other_model in other_models:
                                for other_tag in [tag_2]: # only tag_2!!
                                    other_q_model = copy.deepcopy(other_model)
                                    quantize_model(other_q_model, other_tag, layers_to_ignore=[])
                                    other_q_model.eval()
                                    other_q_models.append(other_q_model)
                                    other_q_model_tags.append(other_tag)
                            other_q_model = None
                            # Then we need to quantize the model and then call find_best_transferable_border_image and communicate with rank 0 in there
                            find_best_transferable_border_image(
                                rank,
                                border_img,
                                other_q_models,
                                [tag_1,tag_2],
                                device,
                                eval_batch_size,
                                src_img.shape[0]
                            )
                        break
                total_num_iters += num_iters_per_img

            if try_idx >= num_tries_per_model - 1:
                #del model_1
                #del model_2
                #raise RuntimeError(
                #    f"❌ Failed to find border image for {i+1}/{total_num_pairs}-th model pair: {tag_1} vs {tag_2}. Please tune the hyperparameters."
                #)
                logger.info(
                    f"❌ Failed to find border image for {i+1}/{total_num_pairs}-th model pair: {tag_1} vs {tag_2}. Please tune the hyperparameters."
                )
                break
        # Cleanup
        model_1.cpu()
        model_2.cpu()
        del model_1
        del model_2
    dist.barrier()

    return border_img_group

def send_dict(dictionary, dst, device):
    # Serialize the dictionary using torch.save
    buffer = io.BytesIO()
    torch.save(dictionary, buffer)
    
    # Convert buffer into a tensor
    buffer.seek(0)
    byte_data = buffer.read()
    byte_tensor = torch.ByteTensor(list(byte_data)).to(device)
    
    # Send the size of the byte_tensor first
    size_tensor = torch.LongTensor([byte_tensor.size(0)]).to(device)
    dist.send(size_tensor, dst=dst)

    # Then send the actual byte tensor
    dist.send(byte_tensor, dst=dst)

def recv_dict(src, device):
    # Receive the size of the incoming byte tensor
    size_tensor = torch.LongTensor([0]).to(device)
    dist.recv(size_tensor, src=src)
    buffer_size = size_tensor.item()
    
    # Receive the actual byte tensor
    byte_tensor = torch.ByteTensor(buffer_size).to(device)
    dist.recv(byte_tensor, src=src)
    
    # Convert the received byte tensor back into byte stream
    byte_data = byte_tensor.cpu().numpy().tobytes()
    buffer = io.BytesIO(byte_data)
    
    # Deserialize the dictionary from the byte stream
    buffer.seek(0)
    dictionary = torch.load(buffer)
    
    return dictionary


def load_ckpt_by_value(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    ckpt: dict[str, torch.Tensor] = torch.load(ckpt_path,map_location=device)
    ori_state_dict = model.state_dict()
    assert list(ori_state_dict.keys()) == list(ckpt.keys()), "Model structure mismatch"
    assert all(ori_state_dict[k].shape == v.shape for k, v in ckpt.items()), "Model param shape mismatch"

    renamed_ckpt = {}
    for k, v in zip(ori_state_dict.keys(), ckpt.values()):
        renamed_ckpt[k] = v

    model.load_state_dict(renamed_ckpt)


def get_dataset(name, transform):
    match name:
        case "cifar10":
            return get_cifar10(transform)
        case "cifar100":
            return get_cifar100(transform)
        case "imagenet":
            return get_imagenet(transform)
        case _:
            raise ValueError(f"Unknown dataset name: {name}")


def build_a_border_img_meta(border_image: dict, index: int):
    """
    Extract a single border image from a batch of border images
    """
    return dict(
        border_image=border_image["border_image"][index, ...].unsqueeze(0),
        ground_truth_label=border_image["ground_truth_label"][index],
        model_1_tag=border_image["model_1_tag"],
        model_2_tag=border_image["model_2_tag"],
        model_1_pred_label=border_image["model_1_pred_label"][index],
        model_2_pred_label=border_image["model_2_pred_label"][index],
    )


@torch.no_grad
def find_best_transferable_border_image(
    rank: int, border_image: dict, models: list[torch.nn.Module], model_tags: list[str], device, eval_batch_size, train_batch_size
):
    model_tags = [model_tags[0] + f"_g1_{train_batch_size}", model_tags[1] + f"_g2_{train_batch_size}"]
    my_tag = [model_tags[rank]]
    border_img_indices = [None]  # single border image
    if isinstance(border_image["ground_truth_label"], (tuple, list)):
        # image border batch
        is_border_img = border_image["is_border_image"]
        border_img_indices = []
        for i, is_border_img_i in enumerate(is_border_img):
            if is_border_img_i:
                border_img_indices.append(i)

    find_any_fully_transferable = False
    best_transferable_ratio = 0.0
    best_transferable_img = None

    for border_img_idx in border_img_indices:
        if border_img_idx is None:
            border_img = border_image
        else:
            border_img = build_a_border_img_meta(border_image, border_img_idx)
        identifier = DeviceIdentifierOneVsOne([border_img])

        exact_match_list = []
        pred_q_tags = []
        # only this part I need to change
        for q_tag, q_model in zip(my_tag, models):
            q_model.eval()
            q_model = q_model.to(device)
            pred_q_tag = identifier.predict(q_model, eval_batch_size)
            pred_q_tags.append(pred_q_tag)
            exact_match_list.append(pred_q_tag == q_tag)
            q_model = q_model.cpu()
        if rank == 0:
            # Receive the lists from rank 1
            recv_exact_match_list = recv_dict(src=1, device=device)
            recv_pred_q_tags = recv_dict(src=1, device=device)
            print("Exact_match_list:", exact_match_list)
            print("Pred_q_tags:", pred_q_tags)
        
            # Extend the local lists with the received lists from Rank 1
            exact_match_list.extend(recv_exact_match_list)
            pred_q_tags.extend(recv_pred_q_tags)
            
            print("Extended exact_match_list:", exact_match_list)
            print("Extended pred_q_tags:", pred_q_tags)
        elif rank ==1:
            print("Exact_match_list:", exact_match_list)
            print("Pred_q_tags:", pred_q_tags)
            # Send the lists to rank 0
            send_dict(exact_match_list, dst=0, device=device)
            send_dict(pred_q_tags, dst=0, device=device)
            done_tensor = torch.zeros(1, dtype=torch.bool).to(device)
            dist.recv(tensor=done_tensor, src=0)
            if done_tensor.item():
                break
            continue

        transferable_ratio = sum(exact_match_list) / len(exact_match_list)

        # make the model tags actually distinguish between GPU1 and GPU2 for later
        if transferable_ratio >= best_transferable_ratio:
            best_transferable_ratio = transferable_ratio
            best_transferable_img = {
                "border_image_idx": border_img_idx,
                "q_tags": model_tags,
                "pred_q_tags": pred_q_tags,
                "exact_match": exact_match_list,
                "all_match": all(exact_match_list),
                "transferable_ratio": transferable_ratio,
            }

        if transferable_ratio == 1.0:
            find_any_fully_transferable = True
            done_tensor = torch.ones(1, dtype=torch.bool).to(device)
            print("Send we are done")
            dist.send(tensor=done_tensor, dst=1)
            break
        else:
            done_tensor = torch.zeros(1, dtype=torch.bool).to(device)
            print("Send we are not done yet")
            dist.send(tensor=done_tensor, dst=1)

    return find_any_fully_transferable, best_transferable_img


def create_models_from_checkpoints(model_names, model_ckpts, num_classes,device) -> list[torch.nn.Module]:
    other_models = []
    for model_name, model_ckpt in zip(model_names, model_ckpts):
        model, _ = create_model(model_name, num_classes)
        load_ckpt_by_value(model, model_ckpt,device)
        model.cpu()
        model.eval()
        other_models.append(model)
    return other_models


def calculate_current_linear_lr(start_lr, end_lr, total_steps, current_step):
    return start_lr + (end_lr - start_lr) * current_step / total_steps


if __name__ == "__main__":
    set_logging_verbosity(logging.INFO)

    parser = ArgumentParser(
        prog="python img_cls_emulated_batched-GPUs.py",
        description="Engineer a batch of border images for image classification models",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        required=True,
        help="IP address of the master node for torch.distributed init_method"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        required=True,
        help="Port of the master node for torch.distributed init_method"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default = 0,
        help="Rank of server",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "vgg16",
            "resnet18",
            "resnet50",
            "efficientnet_b0",
            "densenet121",
            "mobilenet_v2",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
        ],
        default = "resnet18",#"resnet50",
        help="Model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "imagenet"],
        help="Image classification dataset name",
        default = "cifar10",
    )
    parser.add_argument(
        "--save_dir",
        "--save-dir",
        dest="save_dir",
        type=str,
        help="Directory to save the fine-tuned model, border images, and logs",
        default=DEFAULT_SAVE_DIR,
    )
    parser.add_argument(
        "--model-ckpt",
        dest="model_ckpt",
        type=str,
        default=DEFAULT_CKPT,#"checkpoints/default/resnet18-cifar10.pt",
        help="Model checkpoint path. If not provided, a fp32 model will be created and fine-tuned.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["1-vs-1-pgd", "1-vs-rest"],
        default="1-vs-1-pgd",
        help="Method to engineer border images",
    )
    parser.add_argument(
        "--q-tags",
        dest="q_tags",
        nargs="+",
        type=str,
        default=["fp32", "bf16", "fp16"]#, "mxint8", "fp8-e3m4", "fp8-e4m3", "int8-dynamic"],#["fp32", "bf16"],
    )
    
    parser.add_argument(
        "--num-iters",
        dest="num_iters",
        type=int,
        default=1000,
        help="Number of iterations to engineer border images",
    )
    parser.add_argument(
        "--num-tries",
        dest="num_tries",
        type=int,
        default=3,#80,
        help="Number of tries to engineer border images for each q_tag combination",
    )
    parser.add_argument(
        "--noise-scale", dest="noise_scale", type=float, default=0.01, help="Noise scale to add to the source image"
    )#0.01
    # parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for engineering border images")
    # parser.add_argument("--lr-decay", type=float, default=1.0, help="Learning rate decay factor")
    # parser.add_argument("--lr-decay-every", type=int, default=50, help="Learning rate decay every N iterations")
    parser.add_argument(
        "--start-lr", type=float, default=1e-3, help="Start learning rate for engineering border images"
    )#1e-3, 5e-3,
    parser.add_argument("--end-lr", type=float, default=1e-5, help="End learning rate for engineering border images")#1e-4, 5e-4

    parser.add_argument(
        "--check-every",
        type=float,
        default=100,#50#None,
        help="Check border image every N iterations. If less than 1.0, it is treated as a ratio.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    parser.add_argument(
        "--fine-tune-batch-size",
        dest="fine_tune_batch_size",
        type=int,
        default=32,#32,#128,
        help="Batch size for fine-tuning model",
    )
    parser.add_argument(
        "--num-workers", dest="num_workers", type=int, default=8, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--skip-q-test",
        dest="skip_q_test",
        action="store_true",
        help="Engineer border images without testing quantized models",
    )
    parser.add_argument(
        "--engineer_batch_sizes",
        dest="engineer_batch_sizes",
        nargs="+",
        type=int,
        default=[32],#[16,32],#[1,2,4,16,32,64,128,256,512,1024],
        help="Batch sizes for engineering border images",
    )
    parser.add_argument(
        "--eval-batch-size",
        dest="eval_batch_size",
        type=int,
        default=32, #32
        help="Batch size for evaluating border images",
    )
    parser.add_argument(
        "--other-models",
        dest="other_models",
        type=str,
        nargs="+",
        default=["resnet18"],#None,["resnet50"],
        help="Other models to check the transferability of the border images",
    )
    parser.add_argument(
        "--other-model-ckpts",
        dest="other_model_ckpts",
        type=str,
        nargs="+",
        default=[OTHER_CKPT],#None,
        help="Model checkpoints of the other models to check the transferability of the border images",
    )
    parser.add_argument(
        "--other-model-tags",
        type=str,
        nargs="+",
        default=None,
        help="Tags of the other models to check the transferability of the border images. If None, --q-tags will be used.",
    )
    parser.add_argument(
        "--create-model-ckpt-only",
        action="store_true",
        help="Create model checkpoint only without engineering border images",
    )

    args = parser.parse_args()
    logger.info(f"Arguments:\n{pformat(vars(args), sort_dicts=False)}")

    set_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # ====================== Load dataset and create model =================================
    num_classes = TASK_TO_NUM_CLASSES[args.dataset]
    model, transform = create_model(args.model_name, num_classes)
    train_dataset, val_dataset, test_dataset = get_dataset(args.dataset, transform)
    # fmt: off
    train_dataloader = DataLoader(train_dataset, batch_size=args.fine_tune_batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.fine_tune_batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.fine_tune_batch_size, shuffle=False, num_workers=args.num_workers)
    

    if args.model_ckpt is not None:
        load_ckpt_by_value(model, args.model_ckpt, device)
        logger.info(f"Loaded model checkpoint from: {args.model_ckpt}")
    else:
        fine_tune_model(model, train_dataloader, val_dataloader, test_dataloader, save_dir, device=device)
        if args.create_model_ckpt_only:
            logger.info(f"--create_model_ckpt_only enabled. Model checkpoint is saved at: {save_dir}")
            exit(0)

    model.cpu()

    # ====================== Quantization Tags =========================================
    q_tags = copy.deepcopy(args.q_tags)
    assert len(set(q_tags)) == len(q_tags), f"Duplicate quantization tags {q_tags}"

    # ====================== Check accuracy of quantized models ======================
    test_acc = None
    world_size = 2  # Number of GPUs (one on each server)
    devices = [0, 0]  # First GPU on both servers (CUDA device IDs)
    setup_distributed(
        rank=args.rank,
        world_size=world_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
    )
    if not args.skip_q_test:
        run_quantization_and_testing(args.rank, world_size, model, q_tags, test_dataloader,devices)
    q_model = None

    # ====================== Engineer border images =================================
    if args.check_every is None:
        check_every = args.num_iters
    elif 0 < args.check_every < 1.0:
        check_every = math.ceil(args.check_every * args.num_iters)
    elif args.check_every >= 1.0:
        check_every = int(args.check_every)
    else:
        raise ValueError(f"Invalid check_every: {args.check_every}")

    # create other models for checking transferability
    other_models = None
    other_model_tags = None
    if args.other_models is not None:
        assert args.other_model_ckpts is not None, "Please provide model checkpoints for the other models"
        if args.other_model_tags is not None:
            other_model_tags = args.other_model_tags
        else:
            other_model_tags = q_tags
        other_models = create_models_from_checkpoints(args.other_models, args.other_model_ckpts, num_classes,device)
    # Perhaps this part can be repeated for different batch sizes?
    border_img_group_list = []
    for batch_size in args.engineer_batch_sizes:
        print("Start engineering border images for batch size: ",batch_size)
        src_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
        if args.eval_batch_size!=1:
            args.eval_batch_size = batch_size
        if args.method == "1-vs-1-pgd":
            border_img_group_ = main_worker(args.rank,world_size,model=model,
                model_tags=q_tags,
                dataloader=src_dataloader,
                noise_scale=args.noise_scale,
                # lr=args.lr,
                # lr_decay=args.lr_decay,
                # lr_decay_every=args.lr_decay_every,
                start_lr=args.start_lr,
                end_lr=args.end_lr,
                num_iters_per_img=args.num_iters,
                num_tries_per_model=args.num_tries,
                check_every=check_every,
                devices=devices,
                other_models=other_models, 
                other_tags=other_model_tags,
                eval_batch_size = args.eval_batch_size)
            if args.rank == 0:
                border_img_group_list.extend(border_img_group_)
        elif args.method == "1-vs-rest":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown method: {args.method}")
        dist.barrier()

    # ====================== Check transferability of all border images ===================
    
    device = devices[args.rank]
    if args.rank == 0:
        border_img_group = border_img_group_list
        send_dict(border_img_group_list, dst=1, device=device)
    elif args.rank == 1:
        border_img_group_list = recv_dict(src=0, device=device)
        border_img_group = border_img_group_list
    # extend q_tags and other_model_tags to tags of each batch size
    q_tags_g1, q_tags_g2, other_model_tags_g1, other_model_tags_g2 = [], [], [], []
    for batch_size in args.engineer_batch_sizes:
        for tag in q_tags:
            q_tags_g1.append(tag+f"_g1_{batch_size}")
            q_tags_g2.append(tag+f"_g2_{batch_size}")
        for tag in other_model_tags:
            other_model_tags_g1.append(tag+f"_g1_{batch_size}")
            other_model_tags_g2.append(tag+f"_g2_{batch_size}")
    if args.rank == 0:
        other_model_tags = other_model_tags_g1
        q_tags = q_tags_g1
    else:
        other_model_tags = other_model_tags_g2
        q_tags = q_tags_g2
    if other_models is not None:
        border_img_group = []
        # In border_img_group I should have all the best border images for different quantization combinations and two different gpus and different batch sizes
        for border_img in border_img_group_list:
            best_idx = border_img["best_transferable_image"]["border_image_idx"]
            border_img_group.append(build_a_border_img_meta(border_img, best_idx))

        identifier = DeviceIdentifierOneVsOne(border_img_group)
        exact_match = {"by_tag": {}, "by_model": {}}
        for other_m_name, other_model in zip(args.other_models, other_models):
            for other_q_tag in other_model_tags:
                print("Other q tag: ",other_q_tag)
                if other_q_tag not in exact_match["by_tag"]:
                    exact_match["by_tag"][other_q_tag] = []
                if other_m_name not in exact_match["by_model"]:
                    exact_match["by_model"][other_m_name] = []
                other_q_model = copy.deepcopy(other_model)
                other_q_model.eval()
                quantize_model(other_q_model, other_q_tag.split('_')[0], layers_to_ignore=[])
                other_q_model.to(device)
                if args.eval_batch_size!=1:
                    args.eval_batch_size = int(other_q_tag.split('_')[2])
                pred_q_tag = identifier.predict(other_q_model, args.eval_batch_size)
                print("Pred tag: ",pred_q_tag)
                other_q_model.cpu()
                exact_match["by_tag"][other_q_tag].append(pred_q_tag == other_q_tag)
                exact_match["by_model"][other_m_name].append(pred_q_tag == other_q_tag)
        other_model = None
        if args.rank ==1:
            print("Send exact match to rank 0 ")
            send_dict(exact_match, dst=0, device=device)
        elif args.rank == 0:
            print("Exact match: ",exact_match)
            exact_match2 = recv_dict(src=1, device=device)
            print("Exact match 2: ",exact_match2)
            print("Other model tags: ",other_model_tags)
            exact_matches = []
            for q_tag in other_model_tags_g1:
                exact_matches.extend(exact_match["by_tag"][q_tag])
            print(exact_matches)
            for q_tag in other_model_tags_g2:
                if q_tag in exact_match2["by_tag"]:
                    exact_matches.extend(exact_match2["by_tag"][q_tag])
            print(exact_matches)
            overall_accuracy = sum(exact_matches) / len(exact_matches)
            
            by_tag_accuracy = {
                q_tag: sum(exact_match["by_tag"][q_tag]) / len(exact_match["by_tag"][q_tag]) for q_tag in other_model_tags_g1
            }
            for q_tag in other_model_tags_g2:
                if q_tag in exact_match2["by_tag"]:
                    by_tag_accuracy[q_tag] = (
                        sum(exact_match2["by_tag"][q_tag]) / len(exact_match2["by_tag"][q_tag])
                    )
            print("by_tag_accuracy: ",by_tag_accuracy)
            # Calculate accuracy by model
            by_model_accuracy = {}
            for m_name in args.other_models:
                # Get statistics from both exact_match and exact_match2
                match_list = exact_match["by_model"].get(m_name, []) + exact_match2["by_model"].get(m_name, [])
                if match_list:
                    by_model_accuracy[m_name] = sum(match_list) / len(match_list)
                else:
                    by_model_accuracy[m_name] = 0
            print("by_model_accuracy: ",by_model_accuracy)
            logger.info(f"Transferability accuracy by q_tag:\n{pformat(by_tag_accuracy, sort_dicts=False)}")
            logger.info(f"Transferability accuracy by model:\n{pformat(by_model_accuracy, sort_dicts=False)}")
            logger.info(
                f"🔄 Overall transferability accuracy: {overall_accuracy:.4f} (random_guess={1/len(other_model_tags):.4f})"
            )
    cleanup_distributed()
    # ====================== Save border images ======================================
    if args.rank == 0:
        border_img_dir = save_dir.joinpath("border_image_group")
        border_img_dir.mkdir(parents=True, exist_ok=True)
    
        for i, border_img_dict in enumerate(border_img_group):
            border_img = border_img_dict.pop("border_image")  # (B, C, H, W)
            border_img_path = border_img_dir.joinpath(f"border_img_{i}.pt")
            torch.save(border_img, border_img_path)
            border_img_dict["border_image_path"] = border_img_path.resolve().as_posix()
            # save as png
            tv.utils.save_image(border_img, border_img_path.with_suffix(".png"))
    
        border_img_group_yaml = save_dir.joinpath("border_image_group.yaml")
        with border_img_group_yaml.open("w") as f:
            yaml.safe_dump(border_img_group, f)
    
        if other_models is not None:
            transferability_path = save_dir.joinpath("transferability.yaml")
            with open(transferability_path, "w") as f:
                yaml.safe_dump(
                    dict(
                        exact_match=exact_match,
                        overall_accuracy=overall_accuracy,
                        by_tag_accuracy=by_tag_accuracy,
                        by_model_accuracy=by_model_accuracy,
                    ),
                    f,
                )
    
        if test_acc is not None:
            test_acc_path = save_dir.joinpath("q_test_acc.yaml")
            with test_acc_path.open("w") as f:
                yaml.safe_dump(test_acc, f)
    
        # save args
        args_path = save_dir.joinpath("args.yaml")
        with args_path.open("w") as f:
            yaml.safe_dump(vars(args), f)
