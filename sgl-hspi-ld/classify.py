import logging
from pathlib import Path
from pprint import pformat
import json
import time
import datetime
from dataclasses import dataclass, asdict
import pickle
import fnmatch

import numpy as np
from huggingface_hub import HfApi
from transformers import set_seed
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

logger = logging.getLogger(__name__)


def download(
    repo_id: str = "AnonymousPineapple98/HSPI-SGL",
    save_dir: str = "data",
    token: str = None,
):
    api = HfApi()
    local_path = api.snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        token=token,
    )
    return local_path


def find_dataset_dir(dataset_root: Path, dataset_dirs: list[Path]):
    if dataset_root.is_dir():
        if dataset_root.joinpath("env_info.txt").exists():
            dataset_dirs.append(dataset_root)
        else:
            for child in dataset_root.iterdir():
                find_dataset_dir(child, dataset_dirs)
    else:
        return None


def split_fp32(x: np.ndarray) -> np.ndarray:
    """
    Split FP32 into sign, exponent, mantissa

    shape: [...] -> [..., 3]
    """
    assert x.dtype == np.float32
    x = x.view(np.int32)
    sign = (x >> 31).astype(np.float32)
    exp = ((x >> 23) & 0xFF).astype(np.float32)
    mantissa = (x & 0x7FFFFF).astype(np.float32)

    stacked = np.stack([sign, exp, mantissa], axis=-1)
    return stacked


def filter_tags(tags: list[str], patterns: list[str]):
    filtered_tags = []
    for tag in tags:
        for pattern in patterns:
            if fnmatch.fnmatch(tag, pattern):
                filtered_tags.append(tag)
                break
    return filtered_tags


def load_logit_datasets(
    logits_datasets: list[Path],
    truncate_first_n_tokens: int = None,
    truncate_first_n_logits: int = None,
    q_tags_to_remove: list[str] = None,
    q_tags_to_keep: list[str] = None,
    q_tag_transform: dict[str, str] = None,
):
    if (q_tags_to_remove is not None) and (q_tags_to_keep is not None):
        raise ValueError("Cannot specify both q_tags_to_remove and q_tags_to_keep")

    # idx.shape = [n_batches]
    # top_logits.shape = [n_batches, batch_size, n_tokens, n_logits]

    logits_list = []
    label_idx_list = []
    label_to_idx = {}
    labels = []

    for dataset_dir_i in logits_datasets:
        dataset_path_i = Path(dataset_dir_i)
        assert dataset_path_i.is_dir(), f"{dataset_path_i} is not a directory"
        with open(dataset_path_i / "tag2idx.json", "r") as f:
            label_to_idx_i = json.load(f)
        labels.extend(label_to_idx_i.keys())

    labels = sorted(list(set(labels)))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    print(f"Merged label_to_idx:\n{pformat(label_to_idx, sort_dicts=False)}")

    for dataset_dir_i in logits_datasets:
        dataset_path_i = Path(dataset_dir_i)
        logits_i = np.load(dataset_path_i / "top_logits.npy")
        if truncate_first_n_tokens is not None:
            logits_i = logits_i[:, :, :truncate_first_n_tokens, :]
        if truncate_first_n_logits is not None:
            logits_i = logits_i[:, :, :, :truncate_first_n_logits]
        label_idx_i = np.load(dataset_path_i / "idx.npy")
        assert (
            label_idx_i.shape[0] == logits_i.shape[0]
        ), f"Number of samples mismatch: {label_idx_i.shape[0]} != {logits_i.shape[0]}"

        with open(dataset_path_i / "tag2idx.json", "r") as f:
            label_to_old_idx_i = json.load(f)

        old_idx_to_label_i = {i: label for label, i in label_to_old_idx_i.items()}

        @np.vectorize
        def update_idx(old_idx):
            return label_to_idx[old_idx_to_label_i[old_idx]]

        label_idx_i = update_idx(label_idx_i)

        logits_list.append(logits_i)
        label_idx_list.append(label_idx_i)
        print(f"Loaded logits from {dataset_path_i}")

    if truncate_first_n_tokens is None:
        truncate_first_n_tokens = min([logits.shape[2] for logits in logits_list])
        logits_list = [
            logits[:, :, :truncate_first_n_tokens, :] for logits in logits_list
        ]
        logger.warning(
            f"Automatically truncated first {truncate_first_n_tokens} tokens as `truncate_first_n_tokens` is not specified"
        )
    if truncate_first_n_logits is None:
        truncate_first_n_logits = min([logits.shape[3] for logits in logits_list])
        logits_list = [
            logits[:, :, :truncate_first_n_logits, :] for logits in logits_list
        ]
        logger.warning(
            f"Automatically truncated first {truncate_first_n_logits} logits as `truncate_first_n_logits` is not specified"
        )

    logits = np.concatenate(logits_list, axis=0)
    label_idx = np.concatenate(label_idx_list, axis=0)

    if q_tags_to_remove is not None:
        print(f"🚧 Removing quantizer tags {q_tags_to_remove}")
        q_tags_to_remove = filter_tags(label_to_idx.keys(), q_tags_to_remove)
        q_tag_idx_to_remove = [label_to_idx[q_tag] for q_tag in q_tags_to_remove]
        mask = np.isin(label_idx, q_tag_idx_to_remove, invert=True)
        logits = logits[mask]
        label_idx = label_idx[mask]
        for q_tag in q_tags_to_remove:
            del label_to_idx[q_tag]

        new_label_to_idx = {label: idx for idx, label in enumerate(label_to_idx.keys())}

        @np.vectorize
        def update_idx(old_idx):
            return new_label_to_idx[idx_to_label[old_idx]]

        label_idx = update_idx(label_idx)
        label_to_idx = new_label_to_idx
        idx_to_label = {v: k for k, v in new_label_to_idx.items()}
        print(f"Removed quantizer tags {q_tags_to_remove}")
        print(f"Updated label_to_idx:\n{pformat(label_to_idx, sort_dicts=False)}")
    elif q_tags_to_keep is not None:
        print(f"🚧 Only keeping quantizer tags {q_tags_to_keep}")
        q_tags_to_keep = filter_tags(label_to_idx.keys(), q_tags_to_keep)
        q_tag_idx_to_keep = [label_to_idx[q_tag] for q_tag in q_tags_to_keep]
        mask = np.isin(label_idx, q_tag_idx_to_keep)
        logits = logits[mask]
        label_idx = label_idx[mask]
        q_tags_to_remove = [
            q_tag for q_tag in label_to_idx.keys() if q_tag not in q_tags_to_keep
        ]
        for q_tag in q_tags_to_remove:
            del label_to_idx[q_tag]

        new_label_to_idx = {label: idx for idx, label in enumerate(label_to_idx.keys())}

        @np.vectorize
        def update_idx(old_idx):
            return new_label_to_idx[idx_to_label[old_idx]]

        label_idx = update_idx(label_idx)
        label_to_idx = new_label_to_idx
        idx_to_label = {v: k for k, v in new_label_to_idx.items()}
        print(f"Only keeping quantizer tags {q_tags_to_keep}")
        print(f"Updated label_to_idx:\n{pformat(label_to_idx, sort_dicts=False)}")

    if q_tag_transform is not None:
        # map old tag to new tag
        print(f"🚧 Transforming quantizer tags {q_tag_transform}")
        old_idx_to_old_label = idx_to_label
        old_labels = list(old_idx_to_old_label.values())
        new_labels = []
        for old_label in old_labels:
            if old_label in q_tag_transform:
                new_label = q_tag_transform[old_label]
                new_labels.append(new_label)
            else:
                new_labels.append(old_label)
        new_labels = list(set(new_labels))
        new_label_to_idx = {label: idx for idx, label in enumerate(new_labels)}
        new_idx_to_label = {idx: label for label, idx in new_label_to_idx.items()}
        @np.vectorize
        def update_idx(old_idx):
            return new_label_to_idx[q_tag_transform[old_idx_to_old_label[old_idx]]]

        label_idx = update_idx(label_idx)
        label_to_idx = new_label_to_idx
        idx_to_label = new_idx_to_label
        print(f"Updated label_to_idx:\n{pformat(label_to_idx, sort_dicts=False)}")

    unique, counts = np.unique(label_idx, return_counts=True)
    ratio = counts / counts.sum()
    labels_to_ratio = {idx_to_label[k]: round(v, 4) for k, v in zip(unique, ratio)}
    labels_to_count = {idx_to_label[k]: v for k, v in zip(unique, counts)}
    dataset_profile = {}
    for label in labels_to_ratio:
        dataset_profile[label] = (
            f"{labels_to_count[label]} samples ({labels_to_ratio[label]:.2%})"
        )
    print(
        f"Logits dataset loaded ({label_idx.shape[0]} samples in total), label histogram:\n{pformat(dataset_profile, sort_dicts=False, width=240)}"
    )

    logits = np.ascontiguousarray(logits)
    label_idx = np.ascontiguousarray(label_idx)
    return logits, label_idx, label_to_idx, idx_to_label, truncate_first_n_tokens, truncate_first_n_logits


def calculate_metrics(
    y_train: np.ndarray, y_train_pred: np.ndarray, idx_to_label: dict[int, str]
):
    confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
    class_acc = confusion_matrix_train.diagonal() / confusion_matrix_train.sum(axis=1)
    class_acc = {idx_to_label[i]: acc for i, acc in enumerate(class_acc.tolist())}
    print(f"By-class accuracy:\n{pformat(class_acc, sort_dicts=False, width=120)}")

    matthews_corr_train = matthews_corrcoef(y_train, y_train_pred)
    print(f"Training Matthews correlation: {matthews_corr_train:.4f}")

    classification_report_train = classification_report(
        y_train,
        y_train_pred,
        labels=np.unique(y_train),
        target_names=[idx_to_label[i] for i in np.unique(y_train)],
        output_dict=True,
    )
    classification_report_train_txt = classification_report(
        y_train,
        y_train_pred,
        labels=np.unique(y_train),
        target_names=[idx_to_label[i] for i in np.unique(y_train)],
    )
    print(
        f"Training classification report:\n{classification_report_train_txt}"
    )
    return (
        confusion_matrix_train,
        class_acc,
        matthews_corr_train,
        classification_report_train,
    )


@dataclass
class SVMConfig:
    max_iter: int = 1000


def train_svm(
    dataset_root: Path,
    svm_cfg: SVMConfig,
    truncate_first_n_tokens: int = None,
    truncate_first_n_logits: int = None,
    q_tags_to_remove: list[str] = None,
    q_tags_to_keep: list[str] = None,
    q_tag_transform: dict[str, str] = None,
    seed: int = 42,
    save_dir: Path = None,
):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    set_seed(seed)
    if save_dir is None:
        save_dir = Path("svm").joinpath(timestamp)
    args = {
        "dataset_root": str(dataset_root),
        "svm_cfg": asdict(svm_cfg),
        "truncate_first_n_tokens": truncate_first_n_tokens,
        "truncate_first_n_logits": truncate_first_n_logits,
        "q_tags_to_remove": q_tags_to_remove,
        "q_tags_to_keep": q_tags_to_keep,
        "q_tag_transform": q_tag_transform,
        "seed": seed,
        "save_dir": str(save_dir),
    }

    print(f"🚀 Loading datasets...")
    logits_datasets = []
    find_dataset_dir(dataset_root, logits_datasets)
    logits, label_idx, label_to_idx, idx_to_label, truncate_first_n_tokens_, truncate_first_n_logits_ = load_logit_datasets(
        logits_datasets=logits_datasets,
        truncate_first_n_logits=truncate_first_n_logits,
        truncate_first_n_tokens=truncate_first_n_tokens,
        q_tags_to_keep=q_tags_to_keep,
        q_tags_to_remove=q_tags_to_remove,
        q_tag_transform=q_tag_transform,
    )
    logits = logits.astype(np.float32)
    logits = split_fp32(logits)
    logits = logits.reshape(logits.shape[0], -1)

    scalar = StandardScaler()
    X = scalar.fit_transform(logits)
    X_train = X
    y_train = label_idx

    assert len(np.unique(label_idx)) > 1, "Only one class found in the dataset"

    start = time.time()
    print("🚀 Training SVM model...")
    svc = LinearSVC(random_state=seed, max_iter=svm_cfg.max_iter).fit(X_train, y_train)
    end = time.time()
    print(f"Training time: {end - start:.2f}s")

    model_dir = save_dir.joinpath("svm_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir.joinpath("scalar.pkl"), "wb") as f:
        pickle.dump(scalar, f)
    with open(model_dir.joinpath("svm.pkl"), "wb") as f:
        pickle.dump(svc, f)
    with open(model_dir.joinpath("label_to_idx.json"), "w") as f:
        json.dump(label_to_idx, f)
    data_meta = {
        "truncate_first_n_tokens": truncate_first_n_tokens_,
        "truncate_first_n_logits": truncate_first_n_logits_,
        "q_tags_to_remove": q_tags_to_remove,
        "q_tags_to_keep": q_tags_to_keep,
        "q_tag_transform": q_tag_transform,
    }
    with open(model_dir.joinpath("data_meta.json"), "w") as f:
        json.dump(data_meta, f)
    print(f"Model saved to {model_dir}")

    print("🚀 Evaluating SVM model...")
    y_train_pred = svc.predict(X_train)
    acc_train = accuracy_score(y_train, y_train_pred)
    print(f"Train accuracy: {acc_train:.4f}")

    (
        confusion_matrix_train,
        class_acc,
        matthews_corr_train,
        classification_report_train,
    ) = calculate_metrics(
        y_train=y_train,
        y_train_pred=y_train_pred,
        idx_to_label=idx_to_label,
    )
    eval_dir = save_dir.joinpath("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    training_results = {
        "accuracy": acc_train,
        "classification_report": classification_report_train,
        "confusion_matrix": confusion_matrix_train.tolist(),
        "class_accuracy": class_acc,
        "matthews_correlation": matthews_corr_train.item(),
    }
    with open(eval_dir.joinpath("training_results.json"), "w") as f:
        json.dump(training_results, f, indent=4)
    with open(eval_dir.joinpath("args.json"), "w") as f:
        json.dump(args, f, indent=4)
    print(f"Evaluation results saved to {eval_dir}")


def eval_svm(
    dataset_root: Path,
    svm_ckpt: Path,
    truncate_first_n_tokens: int = None,
    truncate_first_n_logits: int = None,
    q_tags_to_remove: list[str] = None,
    q_tags_to_keep: list[str] = None,
    q_tag_transform: dict[str, str] = None,
    save_dir: Path = None,
):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    if save_dir is None:
        save_dir = Path("svm").joinpath(timestamp)

    args = {
        "dataset_root": str(dataset_root),
        "svm_ckpt": str(svm_ckpt),
        "truncate_first_n_tokens": truncate_first_n_tokens,
        "truncate_first_n_logits": truncate_first_n_logits,
        "q_tags_to_remove": q_tags_to_remove,
        "q_tags_to_keep": q_tags_to_keep,
        "q_tag_transform": q_tag_transform,
        "save_dir": str(save_dir),
    }

    assert svm_ckpt.is_dir(), f"{svm_ckpt} is not a directory"
    with open(svm_ckpt.joinpath("data_meta.json"), "r") as f:
        data_meta = json.load(f)
    if truncate_first_n_tokens is None:
        truncate_first_n_tokens = data_meta["truncate_first_n_tokens"]
        print(f"Using `truncate_first_n_tokens` from data meta: {truncate_first_n_tokens}")
    else:
        logger.warning(
            f"Overriding `truncate_first_n_tokens` in data meta with {truncate_first_n_tokens}"
        )
    if truncate_first_n_logits is None:
        truncate_first_n_logits = data_meta["truncate_first_n_logits"]
        print(f"Using `truncate_first_n_logits` from data meta: {truncate_first_n_logits}")
    else:
        logger.warning(
            f"Overriding `truncate_first_n_logits` in data meta with {truncate_first_n_logits}"
        )
    if q_tags_to_remove is None:
        q_tags_to_remove = data_meta["q_tags_to_remove"]
        print(f"Using `q_tags_to_remove` from data meta: {q_tags_to_remove}")
    else:
        logger.warning(
            f"Overriding `q_tags_to_remove` in data meta with {q_tags_to_remove}"
        )
    if q_tags_to_keep is None:
        q_tags_to_keep = data_meta["q_tags_to_keep"]
        print(f"Using `q_tags_to_keep` from data meta: {q_tags_to_keep}")
    else:
        logger.warning(
            f"Overriding `q_tags_to_keep` in data meta with {q_tags_to_keep}"
        )
    if q_tag_transform is None:
        q_tag_transform = data_meta["q_tag_transform"]
        print(f"Using `q_tag_transform` from data meta: {q_tag_transform}")
    else:
        logger.warning(
            f"Overriding `q_tag_transform` in data meta with {q_tag_transform}"
        )

    logits_datasets = []
    find_dataset_dir(dataset_root, logits_datasets)
    logits, label_idx, label_to_idx, idx_to_label, truncate_first_n_tokens_, truncate_first_n_logits_ = load_logit_datasets(
        logits_datasets=logits_datasets,
        truncate_first_n_logits=truncate_first_n_logits,
        truncate_first_n_tokens=truncate_first_n_tokens,
        q_tags_to_keep=q_tags_to_keep,
        q_tags_to_remove=q_tags_to_remove,
        q_tag_transform=q_tag_transform,
    )
    logits = logits.astype(np.float32)
    logits = split_fp32(logits)
    logits = logits.reshape(logits.shape[0], -1)

    with open(svm_ckpt.joinpath("scalar.pkl"), "rb") as f:
        scalar = pickle.load(f)
    with open(svm_ckpt.joinpath("svm.pkl"), "rb") as f:
        svc = pickle.load(f)

    X = scalar.transform(logits)
    y = label_idx

    print("Evaluating SVM model")
    y_pred = svc.predict(X)
    acc = accuracy_score(y, y_pred)

    (
        confusion_matrix,
        class_acc,
        matthews_corr,
        classification_report,
    ) = calculate_metrics(
        y_train=y,
        y_train_pred=y_pred,
        idx_to_label=idx_to_label,
    )

    eval_dir = save_dir.joinpath("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_results = {
        "accuracy": acc,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_accuracy": class_acc,
        "matthews_correlation": matthews_corr.item(),
    }
    with open(eval_dir.joinpath("eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Evaluation results saved to {eval_dir}")

    with open(eval_dir.joinpath("label_to_idx.json"), "w") as f:
        json.dump(label_to_idx, f)

    with open(eval_dir.joinpath("args.json"), "w") as f:
        json.dump(args, f, indent=4)


if __name__ == "__main__":
    from jsonargparse import CLI

    cli_map = {
        "download": download,
        "train-svm": train_svm,
        "eval-svm": eval_svm,
    }

    cli = CLI(cli_map)
