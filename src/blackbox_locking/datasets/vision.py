import torch
import torchvision as tv

from ..utils import ROOT_DIR


def get_cifar10(transform) -> tuple[tv.datasets.CIFAR10, tv.datasets.CIFAR10, tv.datasets.CIFAR10]:
    data_dir = ROOT_DIR.parents[1].joinpath("data")
    train_val_dataset = tv.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = tv.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [0.8, 0.2])
    return train_dataset, val_dataset, test_dataset


def get_cifar100(transform) -> tuple[tv.datasets.CIFAR100, tv.datasets.CIFAR100, tv.datasets.CIFAR100]:
    data_dir = ROOT_DIR.parents[1].joinpath("data")
    train_val_dataset = tv.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = tv.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [0.8, 0.2])
    return train_dataset, val_dataset, test_dataset


def get_imagenet(transform) -> tuple[tv.datasets.ImageNet, tv.datasets.ImageNet, tv.datasets.ImageNet]:
    data_dir = ROOT_DIR.parents[1].joinpath("data").joinpath("imagenet")
    assert data_dir.is_dir(), f"ImageNet dataset not found at {data_dir}. Please create a symlink to the dataset."
    train_val_dataset = tv.datasets.ImageNet(root=data_dir, split="train", download=True, transform=transform)
    test_dataset = tv.datasets.ImageNet(root=data_dir, split="val", download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [0.8, 0.2])
    return train_dataset, val_dataset, test_dataset
