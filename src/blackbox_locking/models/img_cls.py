import torch.nn as nn
import torchvision as tv


def get_resnet18(num_classes: int) -> tuple[nn.Module, tv.transforms.Compose]:
    weights = tv.models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    model = tv.models.resnet18(weights=weights)

    if num_classes is not None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model, transform


def get_resnet50(num_classes: int) -> tuple[nn.Module, tv.transforms.Compose]:
    weights = tv.models.ResNet50_Weights.DEFAULT
    transform = weights.transforms()

    model = tv.models.resnet50(weights=weights)

    if num_classes is not None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model, transform


def get_vgg16(num_classes: int) -> tuple[nn.Module, tv.transforms.Compose]:
    weights = tv.models.VGG16_Weights.DEFAULT
    transform = weights.transforms()

    model = tv.models.vgg16(weights=weights)

    if num_classes is not None:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    return model, transform


def get_efficientnet_b0(num_classes: int) -> tuple[nn.Module, tv.transforms.Compose]:
    weights = tv.models.EfficientNet_B0_Weights.DEFAULT
    transform = weights.transforms()

    model = tv.models.efficientnet_b0(weights=weights)

    if num_classes is not None:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model, transform


def get_densenet121(num_classes: int) -> tuple[nn.Module, tv.transforms.Compose]:
    weights = tv.models.DenseNet121_Weights.DEFAULT
    transform = weights.transforms()

    model = tv.models.densenet121(weights=weights)

    if num_classes is not None:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model, transform


def get_mobilenet_v2(num_classes: int) -> tuple[nn.Module, tv.transforms.Compose]:
    weights = tv.models.MobileNet_V2_Weights.DEFAULT
    transform = weights.transforms()

    model = tv.models.mobilenet_v2(weights=weights)

    if num_classes is not None:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model, transform


def get_mobilenet_v3_small(num_classes: int) -> tuple[nn.Module, tv.transforms.Compose]:
    weights = tv.models.MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()

    model = tv.models.mobilenet_v3_small(weights=weights)

    if num_classes is not None:
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    return model, transform


def get_mobilenet_v3_large(num_classes: int) -> tuple[nn.Module, tv.transforms.Compose]:
    weights = tv.models.MobileNet_V3_Large_Weights.DEFAULT
    transform = weights.transforms()

    model = tv.models.mobilenet_v3_large(weights=weights)

    if num_classes is not None:
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    return model, transform
