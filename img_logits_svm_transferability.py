import sys
import os
import re
from pathlib import Path

import logging
from shutil import rmtree
from argparse import ArgumentParser
from pprint import pformat
import yaml
import copy
from itertools import combinations
import struct

import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
import torchvision as tv
import numpy as np
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC, SVC
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from src.blackbox_locking.models import (
    get_resnet18,
    get_resnet50,
    get_vgg16,
    get_efficientnet_b0,
    get_densenet121,
    get_mobilenet_v2,
    get_mobilenet_v3_small,
    get_mobilenet_v3_large,
)
from src.blackbox_locking.datasets import get_cifar10
from src.blackbox_locking.utils import set_seed
from src.blackbox_locking.loss_fn import one_vs_rest_cross_entropy_loss
from src.blackbox_locking.quantize import quantize_model

logger = logging.getLogger(__name__)

CIFAR10_NUM_CLASSES = 10
DEFAULT_SAVE_DIR = "checkpoints/default"
DEFAULT_CKPT1 = "checkpoints/default/mobilenet-v2-cifar10.pt"
DEFAULT_CKPT2 = "checkpoints/default/efficientnet-b0-cifar10.pt"

def create_model_cifar10(
    model_ckpt: Path, save_dir: Path, num_epochs=3, batch_size=64, lr=1e-3, num_workers=4, device="cuda:1"
) -> torch.nn.Module:
    """
    Fine-tune a pretrained ResNet50 on CIFAR10 dataset

    dtype: FP32
    """
    # create model, datasets, dataloaders
    if "resnet18" in str(model_ckpt):
        model, transform = get_resnet18(num_classes=CIFAR10_NUM_CLASSES)
    elif "vgg16" in str(model_ckpt):
        model, transform = get_vgg16(num_classes=CIFAR10_NUM_CLASSES)
    elif "efficientnet" in str(model_ckpt):
        model, transform = get_efficientnet_b0(num_classes=CIFAR10_NUM_CLASSES)
    elif "densenet" in str(model_ckpt):
        model, transform = get_densenet121(num_classes=CIFAR10_NUM_CLASSES)
    elif "mobilenet-v2" in str(model_ckpt):
        model, transform = get_mobilenet_v2(num_classes=CIFAR10_NUM_CLASSES)
    elif "mobilenetv3small" in str(model_ckpt):
        model, transform = get_mobilenet_v3_small(num_classes=CIFAR10_NUM_CLASSES)
    elif "mobilenetv3large" in str(model_ckpt):
        model, transform = get_mobilenet_v3_large(num_classes=CIFAR10_NUM_CLASSES)
    else:
        model, transform = get_resnet50(num_classes=CIFAR10_NUM_CLASSES)
    model.float()
    train_dataset, val_dataset, test_dataset = get_cifar10(transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model.to(device)

    # fine-tune pretrained resnet50 on cifar10, record the best model, and test the best model in the end
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_accuracy = 0.0

    criterion = torch.nn.functional.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=num_epochs)

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, lr_scheduler, criterion, epoch, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_ckpt)

    model.load_state_dict(torch.load(model_ckpt))

    test_loss, test_accuracy = validate(model, test_loader, criterion, device, description="Test")

    logger.info(f"Best validation loss: {best_val_loss}, accuracy: {best_val_accuracy}")
    logger.info(f"Test loss: {test_loss}, accuracy: {test_accuracy}")

    # log metrics in a yaml file
    metrics = {
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }

    with open(save_dir / f"{str(model_ckpt).removesuffix('.pt')}-metrics.yaml", "w") as f:
        yaml.dump(metrics, f)

    return model, transform
    
def train_one_epoch(model, train_loader, optimizer, lr_scheduler, criterion, num_epochs, device):
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
        prog_bar.set_postfix({"loss": loss.item()})
    lr_scheduler.step()
    avg_loss = running_loss / len(train_loader)
    logger.info(f"Epoch {num_epochs + 1} training loss: {avg_loss}")
    return avg_loss

def validate(model, val_loader, criterion, device, description="Validation", silent=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm.tqdm(val_loader, desc=description, leave=True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    if not silent:
        logger.info(f"Validation loss: {avg_loss}, accuracy: {accuracy}")
    return avg_loss, accuracy

class CustomTensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        return image, label
        
def get_logits(model, val_loader, criterion, device, description="Validation", silent=False):
    model.eval()
    inputs = []
    output_logits = []
    with torch.no_grad():
        for images, labels in tqdm.tqdm(val_loader, desc=description, leave=True, disable=silent):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            inputs.append(images.cpu())
            output_logits.append(outputs.cpu())
    all_inputs = torch.cat(inputs, dim=0)
    all_logits = torch.cat(output_logits, dim=0)
    return all_inputs, all_logits
    
def float_to_bit_list(float_num):
    packed = struct.pack('!f', float_num) 
    bits = struct.unpack('!I', packed)[0]  
    bit_string = f'{bits:032b}'  
    bit_list = [int(bit) for bit in bit_string]
    return torch.tensor(bit_list, dtype=torch.uint8)
    
def apply_float_to_bit_list(tensor):
    flattened = tensor.flatten()
    bit_tensors = [float_to_bit_list(num.item()) for num in flattened]
    bit_tensors = torch.stack(bit_tensors)
    return bit_tensors.view(*tensor.shape, 32)
    
def extract_params_from_filename(filename):
    pattern =r'.*uniform_logits_(\d+)_(\d+)_([^_]+)_(.+)\.pth'
    
    match = re.match(pattern, filename)
    if match:
        num_samples = int(match.group(1))
        model_ckpt_base = match.group(3) #2
        q_tags_str = match.group(4) #3
        q_tags = q_tags_str.split('_')
        return num_samples, model_ckpt_base, q_tags
    else:
        raise ValueError("Filename does not match the expected pattern")

def load_logits(args_logits_file, args_q_tags, args_num_samples, primary_model):
    uniform_logits = None
    uniform_y_labels = None
    args_q_tags_orig = copy.deepcopy(args_q_tags)
    args_num_samples_orig = args_num_samples
    model_ckpt_base_extracted = ""
    if args_logits_file is not None:
        logits_file = Path(args_logits_file).resolve()
        uniform_logits = torch.load(logits_file)
        print(f"Loaded logits from {logits_file}")
        y_labels_file = args_logits_file.replace('uniform_logits', 'uniform_y_labels', 1)
        y_labels_file = Path(y_labels_file).resolve()
        uniform_y_labels = torch.load(y_labels_file)
        print(f"Loaded y_labels from {y_labels_file}")
        args_num_samples, model_ckpt_base_extracted, args_q_tags = extract_params_from_filename(args_logits_file)
    if not primary_model and args_q_tags_orig != args_q_tags and args_num_samples_orig != args_num_samples:
        return None, None, args_num_samples_orig, model_ckpt_base_extracted, args_q_tags_orig
    return uniform_logits, uniform_y_labels, args_num_samples, model_ckpt_base_extracted, args_q_tags
    

def load_model(args_model_ckpt,args_logits_file,args_num_workers,model_ckpt_base_extracted, primary_model):
    if ".pt" not in args_model_ckpt and primary_model:
        args_model_ckpt = DEFAULT_CKPT1
    elif ".pt" not in args_model_ckpt and not primary_model:
        args_model_ckpt = DEFAULT_CKPT2
    model_ckpt = Path(args_model_ckpt).resolve()
    model_dir = model_ckpt.parent
    model_ckpt_base = os.path.splitext(os.path.basename(model_ckpt))[0]
    if args_logits_file is not None:
        if model_ckpt_base!=model_ckpt_base_extracted:
            logger.warning(
                f"Please give a model path with the same name as the logits data path. You may be using logits produced for a different model. {model_ckpt_base},{model_ckpt_base_extracted}"
            )
    if model_ckpt.is_file():
        logger.info(f"Load the model from {model_dir}")
        if "resnet18" in str(model_ckpt):
            model, transform = get_resnet18(num_classes=CIFAR10_NUM_CLASSES)
        elif "resnet50" in str(model_ckpt):
            model, transform = get_resnet50(num_classes=CIFAR10_NUM_CLASSES)
        elif "vgg16" in str(model_ckpt):
            model, transform = get_vgg16(num_classes=CIFAR10_NUM_CLASSES)
        elif "efficientnet" in str(model_ckpt):
            model, transform = get_efficientnet_b0(num_classes=CIFAR10_NUM_CLASSES)
        elif "densenet" in str(model_ckpt):
            model, transform = get_densenet121(num_classes=CIFAR10_NUM_CLASSES)
        elif "mobilenet-v2" in str(model_ckpt):
            model, transform = get_mobilenet_v2(num_classes=CIFAR10_NUM_CLASSES)
        elif "mobilenetv3small" in str(model_ckpt):
            model, transform = get_mobilenet_v3_small(num_classes=CIFAR10_NUM_CLASSES)
        elif "mobilenetv3large" in str(model_ckpt):
            model, transform = get_mobilenet_v3_large(num_classes=CIFAR10_NUM_CLASSES)
        else:
            logger.warning(
                "Please give a model path with a name including 'resnet50', 'vgg16', 'efficientnet', 'densenet', 'mobilenet' or an empty string."
            )
        model.load_state_dict(torch.load(model_ckpt))
    else:
        logger.info(f"Fine-tune a pretrained model and save the best model to {model_ckpt}")
        model, transform = create_model_cifar10(
            model_ckpt=model_ckpt, save_dir=model_dir, device=device, num_workers=args_num_workers
        )
    return model, transform, model_ckpt_base

def return_quantized_models(model,args_q_tags):
    models = []
    q_tags = copy.deepcopy(args_q_tags)
    assert len(set(q_tags)) == len(q_tags), f"Duplicated tags are not allowed: {q_tags}"
    for q_tag in q_tags:
        q_model = quantize_model(copy.deepcopy(model), q_config=q_tag, layers_to_ignore=[])
        models.append(q_model)

    models = [model.to(device) for model in models]
    return models, q_tags

def create_logits_dataset(models, q_tags, uniform_logits, uniform_y_labels, transform, args_num_samples, args_eval_batch_size,args_num_workers,args_save_dir, model_ckpt_base):
    if uniform_logits is None or uniform_y_labels is None:
        
        concatenated_images = torch.load('concatenated_images_5000_10.pth')
        concatenated_images = concatenated_images.to(torch.float32) / 255.0
        concatenated_labels = torch.load('concatenated_labels_5000_10.pth')
        dataset = CustomTensorDataset(concatenated_images, concatenated_labels)
        
        dataloader = DataLoader(dataset, batch_size=args_eval_batch_size, shuffle=False, num_workers=args_num_workers)
        
        criterion = torch.nn.functional.cross_entropy
        uniform_logit_lst = []
        uniform_y_label_lst = []
        idx = 0
        for model, q_tag in zip(models, q_tags):
            inputs, output_logits = get_logits(
                model, dataloader, criterion, device, description=f"Dataloader ({q_tag})", silent=False
            )
            uniform_logit_lst.append(output_logits)
            uniform_y_label_lst.append(torch.full((output_logits.size(0),), idx, dtype=torch.long))
            idx += 1
        uniform_logits = torch.cat(uniform_logit_lst, dim=0)
        uniform_y_labels = torch.cat(uniform_y_label_lst, dim=0)
        print(f'Shape of uniform_logits: {uniform_logits.shape}')
        print(f'Shape of uniform_y_labels: {uniform_y_labels.shape}')
        q_tags_str = "_".join(q_tags)
        logits_path = os.path.join(args_save_dir, f'A100uniform_logits_{args_num_samples}_{args.no_logits}_{model_ckpt_base}_{q_tags_str}.pth')
        labels_path = os.path.join(args_save_dir, f'A100uniform_y_labels_{args_num_samples}_{args.no_logits}_{model_ckpt_base}_{q_tags_str}.pth')
        torch.save(uniform_logits, logits_path)
        torch.save(uniform_y_labels, labels_path)
        print(f'Saved uniform_logits to {logits_path}')
        print(f'Saved uniform_y_labels to {labels_path}')
    return uniform_logits, uniform_y_labels

def convert_logits_to_bits(uniform_logits, uniform_y_labels, args_no_logits):
    # Convert the logits into bits
    bits_list = apply_float_to_bit_list(uniform_logits).numpy()
    print(bits_list.shape)
    assert uniform_logits.shape[0] % args_no_logits == 0, "Number of logits in a set must divide number of samples"

    # Make sets of n logits
    n_blocks = bits_list.shape[0] // args_no_logits
    concatenated_bits_list = bits_list.reshape(n_blocks, -1)
    assert len(uniform_y_labels) == bits_list.shape[0], "Number of labels must match the number of logits"
    y_labels_new = uniform_y_labels[::args_no_logits]
    assert len(y_labels_new) == n_blocks, "Number of labels does not match the number of concatenated"
    return concatenated_bits_list, y_labels_new


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "--model_ckpt1",
        "--model-ckpt1",
        dest="model_ckpt1",
        type=str,
        help="Path of the Resnet50-CIFAR10 model checkpoint. If not provided, fine-tune a pretrained model and save the best model.",
        default="",
    )
    parser.add_argument(
        "--model_ckpt2",
        "--model-ckpt2",
        dest="model_ckpt2",
        type=str,
        help="Path of the Resnet50-CIFAR10 model checkpoint. If not provided, fine-tune a pretrained model and save the best model.",
        default="",
    )
    parser.add_argument(
        "--save_dir",
        "--save-dir",
        dest="save_dir",
        type=str,
        help="Directory to save the numpy array of quantized noise images and their quantization labels.",
        default=DEFAULT_SAVE_DIR,
    )
    parser.add_argument(
        "--logits_file1",
        "--logits-file1",
        dest="logits_file1",
        type=str,
        help="File where data is saved in numpy arrays of quantized noise images.",
        default=None,
    )
    parser.add_argument(
        "--logits_file2",
        "--logits-file2",
        dest="logits_file2",
        type=str,
        help="File where data is saved in numpy arrays of quantized noise images.",
        default=None,
    )
    parser.add_argument(
        "--q-tags",
        dest="q_tags",
        type=str,
        nargs="+",
        help="Quantization tags for the model. q_tag can be one of ['fp32', 'fp16', 'bf16', 'int8-dynamic', 'fp8-e4m3', 'fp8-e3m4', 'mxint8', 'bm8', 'bl8', 'log8', 'bypass'].",
        # default=["fp8-e4m3", "bf16"],
        default=["fp32", "bf16", "fp16"],#["fp32", "bf16", "fp16", "mxint8", "fp8-e3m4", "fp8-e4m3", "int8-dynamic"],
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_samples", type=int, default=5000, help="How many samples per quantization class to produce.")
    parser.add_argument("--no_logits", type=int, default=10, help="How many samples to put into one set to predict which quantization level with the distribution of logits.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the model.")
    parser.add_argument(
        "--eval-batch-size",
        dest="eval_batch_size",
        type=int,
        default=64,
        help="Batch size for evaluating/training the model.",
    )
    parser.add_argument(
        "--num-workers", dest="num_workers", type=int, default=8, help="Number of workers for dataloader."
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # ================================ Load logits if they exist ================================
    uniform_logits1,uniform_y_labels1, args.num_samples, model_ckpt_base_extracted1, args.q_tags = load_logits(args.logits_file1,args.q_tags,args.num_samples,primary_model=True)
    uniform_logits2,uniform_y_labels2, args.num_samples, model_ckpt_base_extracted2, args.q_tags = load_logits(args.logits_file2,args.q_tags,args.num_samples,primary_model=False)
    # ================================ Load or fine-tune a model ================================
    model1,transform1, model_ckpt_base1 = load_model(args.model_ckpt1,args.logits_file1,args.num_workers,model_ckpt_base_extracted1,primary_model=True)
    model1.eval().cpu()
    model2,transform2, model_ckpt_base2 = load_model(args.model_ckpt2,args.logits_file2,args.num_workers,model_ckpt_base_extracted2,primary_model=False)
    model2.eval().cpu()
    # ================================ Create quantized models and datasets ================================
    models1,q_tags = return_quantized_models(model1,args.q_tags)
    models2, _  = return_quantized_models(model2,args.q_tags)
    # Now we have models, q_tags 
    # ================================ Create uniform noise dataset ================================
    # better to not use transform1 and transform2 for transferability
    uniform_logits1, uniform_y_labels1 = create_logits_dataset(models1, q_tags, uniform_logits1, uniform_y_labels1, transform1, args.num_samples, args.eval_batch_size,args.num_workers,args.save_dir, model_ckpt_base1)
    uniform_logits2, uniform_y_labels2 = create_logits_dataset(models2, q_tags, uniform_logits2, uniform_y_labels2,transform2, args.num_samples, args.eval_batch_size,args.num_workers,args.save_dir, model_ckpt_base2)
    print(uniform_y_labels1)
    print(uniform_y_labels2)
    print("Logits 1: ", uniform_logits1[0:10])
    print("Logits 2: ",uniform_logits2[0:10])
    diff = uniform_logits1[0:10] - uniform_logits2[0:10]
    print(diff)
    print(uniform_logits1)

    # ================================ Create SVM ================================
    concatenated_bits_list1, y_labels_new1 = convert_logits_to_bits(uniform_logits1, uniform_y_labels1, args.no_logits)
    concatenated_bits_list2, y_labels_new2 = convert_logits_to_bits(uniform_logits2, uniform_y_labels2, args.no_logits)
    X_train, X_test, y_train, y_test = concatenated_bits_list1, concatenated_bits_list2, y_labels_new1, y_labels_new2
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svc = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000, penalty='l2', dual='auto')).fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = svc.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Linear SVM: Accuracy: {accuracy} of {model_ckpt_base1} with {args.num_samples} images per quantization class and quantization levels {q_tags} and {args.no_logits} images' logits per set to predict which quantization level with the distribution of logits.")
    logger.info(classification_report(y_test, y_pred))
    
