from argparse import ArgumentParser
from dataclasses import dataclass
import math
import random
from copy import deepcopy
import logging
import time
from pprint import pformat
from pathlib import Path
import yaml
import pickle
import os

import tqdm
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.collect_env import get_pretty_env_info
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
    HqqConfig,
)
from optimum.quanto import QuantizedModelForCausalLM, qint8
from transformers.utils.logging import set_verbosity_error as set_hf_verbosity_error
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef

from blackbox_locking.quantize import quantize_model


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("blackbox_locking." + __name__)


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dim: int
    num_classes: int


class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def train_mlp(mlp_config: MLPConfig, train_loader, val_loader, num_epochs, lr, lr_scheduler_step_size, lr_scheduler_gamma, wandb_run):
    mlp = MLP(mlp_config)
    mlp = mlp.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    optimizer = torch.optim.SGD(mlp.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

    prog_bar = tqdm.tqdm(range(num_epochs), desc="Training MLP", total=num_epochs)

    best_matthews = 0
    best_state_dict = None
    for epoch in prog_bar:
        mlp.train()
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            y_pred = mlp(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        lr_scheduler.step()
        total_loss /= len(train_loader)
        if wandb_run is not None:
            wandb.log({"train/loss": total_loss}, step=epoch)

        mlp.eval()
        y_true, y_pred = [], []
        total_val_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x = x.cuda()
                y = y.cuda()
                y_pred_i = mlp(x).argmax(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(y_pred_i.cpu().numpy())
                loss = criterion(mlp(x), y)
                total_val_loss += loss.item()
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        matthews = matthews_corrcoef(y_true, y_pred)
        total_val_loss /= len(val_loader)

        if wandb_run is not None:
            wandb.log(
                {
                    "val/loss": total_val_loss,
                    "val/acc": acc,
                    "val/f1": f1,
                    "val/matthews": matthews,
                },
                step=epoch,
            )

        if matthews > best_matthews:
            best_matthews = matthews
            best_state_dict = deepcopy(mlp.state_dict())

        prog_bar.set_postfix(
            {
                "train_loss": total_loss,
                "val_loss": total_val_loss,
                "val_acc": acc,
                "val_f1": f1,
                "val_matthews": matthews,
                "val_best_matthews": best_matthews,
            }
        )

    return mlp, best_state_dict


@torch.no_grad()
def test_mlp(mlp: MLP, test_loader, idx_to_label: dict[int, str]):
    # use mlp to predict
    # generate classification report and confusion matrix
    mlp.cuda().eval()
    y_true = []
    y_pred = []

    for i, (x, y) in enumerate(test_loader):
        x = x.cuda()
        y_pred_i = mlp(x).argmax(dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(y_pred_i.cpu().numpy())

    report = classification_report(
        y_true,
        y_pred,
        labels=np.unique(y_true),
        target_names=[idx_to_label[i] for i in np.unique(y_true)],
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    matthews = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return report, cm, accuracy, f1, matthews


class LogitsDataset(Dataset):
    def __init__(self, logits: np.ndarray, labels: np.ndarray) -> None:
        super().__init__()

        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.logits[idx], self.labels[idx]


def load_dataset(dataset_dir: list[str]):
    logits_list = []
    label_idx_list = []
    label_to_idx = {}
    requests = []

    labels = []
    for dataset_dir_i in dataset_dir:
        dataset_path_i = Path(dataset_dir_i)
        assert dataset_path_i.is_dir(), f"{dataset_path_i} does not exist"
        with open(dataset_path_i.joinpath("label_to_idx.yaml"), "r") as f:
            label_to_old_idx_i = yaml.safe_load(f)
        labels.extend(label_to_old_idx_i.keys())

    labels = sorted(list(set(labels)))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    logger.info(f"Merged label_to_idx:\n{pformat(label_to_idx, sort_dicts=False)}")

    for dataset_dir_i in dataset_dir:
        dataset_path_i = Path(dataset_dir_i)
        logits_i = np.load(dataset_path_i.joinpath("logits.npy"))
        label_idx_i = np.load(dataset_path_i.joinpath("idx.npy"))
        with open(dataset_path_i.joinpath("label_to_idx.yaml"), "r") as f:
            label_to_old_idx_i = yaml.safe_load(f)
        with open(dataset_path_i.joinpath("requests.yaml"), "r") as f:
            requests_i = yaml.safe_load(f)
        requests.extend(requests_i)

        old_idx_to_label_i = {v: k for k, v in label_to_old_idx_i.items()}

        def update_idx(old_idx):
            return label_to_idx[old_idx_to_label_i[old_idx]]

        update_idx = np.vectorize(update_idx)
        label_idx_i = update_idx(label_idx_i)

        logits_list.append(logits_i)
        label_idx_list.append(label_idx_i)
        logger.info(f"Logits dataset loaded from {dataset_path_i}")

    # check shape[1] of all logits
    assert all(logits.shape[1] == logits_list[0].shape[1] for logits in logits_list)

    logits = np.concatenate(logits_list, axis=0)
    label_idx = np.concatenate(label_idx_list, axis=0)

    unique, counts = np.unique(label_idx, return_counts=True)
    ratio = counts / counts.sum()
    labels_to_ratio = {idx_to_label[k]: round(v, 4) for k, v in zip(unique, ratio)}
    labels_to_count = {idx_to_label[k]: v for k, v in zip(unique, counts)}
    dataset_profile = {}
    for label in labels_to_ratio:
        dataset_profile[label] = f"{labels_to_count[label]} samples ({labels_to_ratio[label]:.2%})"
    logger.info(
        f"Logits dataset loaded ({label_idx.shape[0]} samples in total), label distribution:\n{pformat(dataset_profile, sort_dicts=False)}"
    )

    logits_dataset = LogitsDataset(logits, label_idx)
    return logits_dataset, label_to_idx, idx_to_label, requests



if __name__ == "__main__":
    set_hf_verbosity_error()
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", nargs="+", type=str, required=True)
    parser.add_argument("--mlp-hidden-dim", type=int, default=2048)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--lr-scheduler-step-size", type=int, default=5)
    parser.add_argument("--lr-scheduler-gamma", type=float, default=0.8)

    parser.add_argument("--enable-wandb", "-wdb", dest="enable_wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()
    logger.info(f"Arguments: {pformat(vars(args))}")
    transformers.set_seed(args.seed)

    wandb_run = None
    if args.enable_wandb:
        wandb_run = wandb.init(project="blackbox-locking", group="llm_logits_mlp", config=vars(args), name=args.wandb_name)

    output_dir = Path(args.output_dir) if args.output_dir is not None else None

    logger.info("🚀 Loading logits dataset...")
    logits_dataset, label_to_idx, idx_to_label, requests = load_dataset(args.dataset_dir)

    train_idx, test_idx = train_test_split(np.arange(len(logits_dataset)), test_size=0.2, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=args.seed)
    train_dataset = Subset(logits_dataset, train_idx)
    val_dataset = Subset(logits_dataset, val_idx)
    test_dataset = Subset(logits_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)



    logger.info("🚀 Training MLP...")
    mlp_config = MLPConfig(
        input_dim=logits_dataset.logits.shape[1],
        hidden_dim=args.mlp_hidden_dim,
        num_classes=len(label_to_idx),
    )
    logger.info(f"MLP config:\n{pformat(mlp_config)}")

    mlp, best_state_dict = train_mlp(
        mlp_config=mlp_config,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lr_scheduler_step_size=args.lr_scheduler_step_size,
        lr_scheduler_gamma=args.lr_scheduler_gamma,
        wandb_run=wandb_run,
    )
    logger.info("🚀 Testing MLP...")
    if best_state_dict is not None:
        logger.info("Loading best state dict...")
        mlp.load_state_dict(best_state_dict)
    if output_dir is not None:
        mlp_ckpt_path = output_dir.joinpath("mlp_model").joinpath("best.ckpt")
        mlp_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        mlp.cpu()
        torch.save(mlp.state_dict(), mlp_ckpt_path)

    report, cm, test_acc, test_f1, test_matthews= test_mlp(mlp, test_loader, idx_to_label)
    logger.info(f"Matthews correlation coefficient: {test_matthews}")
    logger.info(f"Classification report:\n{pformat(report)}")
    class_acc = cm.diagonal() / cm.sum(axis=1)
    logger.info(
        f"Class accuracy:\n{pformat({idx_to_label[i]: acc for i, acc in enumerate(class_acc)}, sort_dicts=False)}"
    )
    if wandb_run is not None:
        wandb.log(
            {
                "test/acc": test_acc,
                "test/f1": test_f1,
                "test/matthews": test_matthews,
            }
        )

    # close wandb run
    if args.enable_wandb:
        wandb_run.finish(quiet=True)
