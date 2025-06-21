import logging
from copy import deepcopy
from pprint import pformat
from collections import Counter

import torch
from transformers.modeling_utils import PreTrainedModel

from .qnn import QConv2d, QLinear, quantize_hf_to_builtin_dtype, replace_attn_layers, wrap_forward_with_dtypes
from .utils import (
    set_layer_by_name,
    find_matched_pattern,
)

logger = logging.getLogger(__name__)
logger.propagate = False

DEFAULT_Q_CONFIGS = {
    # built-in:
    "fp32": {"x": {"name": "fp32"}, "w": {"name": "fp32"}},
    "fp16": {"x": {"name": "fp16"}, "w": {"name": "fp16"}},
    "bf16": {"x": {"name": "bf16"}, "w": {"name": "bf16"}},
    # emulated:
    "int8-dynamic": {
        "x": {
            "name": "integer",
            "fp_min": None,
            "fp_max": None,
            "n_bits": 8,
            "is_affine": True,
        },
        "w": {
            "name": "integer",
            "fp_min": None,
            "fp_max": None,
            "n_bits": 8,
            "is_affine": True,
        },
    },
    "fp8-e4m3": {
        "x": {
            "name": "minifloat",
            "width": 8,
            "exponent_width": 4,
            "exponent_bias": None,
        },
        "w": {
            "name": "minifloat",
            "width": 8,
            "exponent_width": 4,
            "exponent_bias": None,
        },
    },
    "fp8-e3m4": {
        "x": {
            "name": "minifloat",
            "width": 8,
            "exponent_width": 3,
            "exponent_bias": None,
        },
        "w": {
            "name": "minifloat",
            "width": 8,
            "exponent_width": 4,
            "exponent_bias": None,
        },
    },
    "mxint8": {
        "x": {"name": "mxint", "width": 8, "block_size": 32, "block_axis": -1},
        "w": {"name": "mxint", "width": 8, "block_size": 32, "block_axis": -1},
    },
    "bm8": {
        "x": {
            "name": "block_minifloat",
            "width": 8,
            "exponent_width": 2,
            "exponent_bias_width": 8,
            "block_size": [16, 16],
            "skip_first_dim": True,
        },
        "w": {
            "name": "block_minifloat",
            "width": 8,
            "exponent_width": 2,
            "exponent_bias_width": 8,
            "block_size": [16, 16],
            "skip_first_dim": False,
        },
    },
    "bl8": {
        "x": {"name": "block_logarithm", "width": 8, "block_size": [16], "skip_first_dim": True},
        "w": {"name": "block_logarithm", "width": 8, "block_size": [16], "skip_first_dim": False},
    },
    "log8": {
        "x": {"name": "logarithm", "width": 8},
        "w": {"name": "logarithm", "width": 8},
    },
    "bypass": {"x": {"name": "bypass"}, "w": {"name": "bypass"}},
}


def build_default_q_config(q_name: str):
    assert q_name in DEFAULT_Q_CONFIGS, f"Unknown quantizer name {q_name}"

    default_q_config = {
        "conv2d": {
            "x": deepcopy(DEFAULT_Q_CONFIGS[q_name]["x"]),
            "w": deepcopy(DEFAULT_Q_CONFIGS[q_name]["w"]),
        },
        "linear": {
            "x": deepcopy(DEFAULT_Q_CONFIGS[q_name]["x"]),
            "w": deepcopy(DEFAULT_Q_CONFIGS[q_name]["w"]),
        },
        "matmul": {
            "x": deepcopy(DEFAULT_Q_CONFIGS[q_name]["x"]),
            "w": deepcopy(DEFAULT_Q_CONFIGS[q_name]["w"]),
        },
    }

    if q_name == "mxint8":
        default_q_config["matmul"]["w"]["block_axis"] = -2

    return default_q_config


def is_builtin_dtype(q_tag: str):
    builtin_types = ["fp32", "fp16", "bf16"]
    for bt in builtin_types:
        if bt in q_tag:
            return True
    return False


def parse_builtin_dtype(q_tag: str):
    builtin_types = ["fp32", "fp16", "bf16"]
    for bt in builtin_types:
        if bt in q_tag:
            return bt
    return None


def quantize_to_built_in_dtype(model: torch.nn.Module, q_tag: str):
    assert is_builtin_dtype(q_tag), f"Unsupported quantizer tag {q_tag}"
    match parse_builtin_dtype(q_tag):
        case "fp32":
            dtype = torch.float32
        case "fp16":
            dtype = torch.float16
        case "bf16":
            dtype = torch.bfloat16
        case _:
            raise ValueError(f"Unknown quantizer tag {q_tag}")

    if not isinstance(model, PreTrainedModel):
        model.to(dtype)
        wrap_forward_with_dtypes(model, dtype, cast_dtype=torch.float32)
    else:
        quantize_hf_to_builtin_dtype(model, dtype)
        wrap_forward_with_dtypes(model, dtype, cast_dtype=torch.float32)


def quantize_model(
    model: torch.nn.Module,
    q_config: dict | str,
    layers_to_ignore: list[str] = [r"lm_head"],
) -> torch.nn.Module:
    """Replace the layers in the model with quantized layers.

    **Note that the activations and weights are quantized at inference runtime**,
    so the model state dict after quantization are still values in full precision instead of quantized values.

    :param model: FP32 model to be quantized.
    :type model: torch.nn.Module
    :param q_config: tag of the quantizer or a dictionary containing the quantizer configuration.
                     Supported quantizer tags: "fp32", "fp16", "bf16", "int8-dynamic", "fp8-e4m3", "fp8-e3m4", "mxint8", "bm8", "bl8", "log8", "bypass".
    :type q_config: dict | str
    :param layers_to_ignore: a list of regex expressions, which are layer name patterns to keep in full precision, defaults to []
    :type layers_to_ignore: list[str], optional
    :return: Quantized model.
    :rtype: torch.nn.Module


    Example:
    >>> import torch
    >>> from torchvision.models import get_model
    >>> from blackbox_locking.quantize import quantize_model
    >>> model = get_model("resnet18", weights="DEFAULT").eval().cuda()
    >>> model_q = quantize_model(model, "mxint8")
    """

    # fp32, bf16, fp16
    if isinstance(q_config, str) and is_builtin_dtype(q_config):
        quantize_to_built_in_dtype(model, q_config)
        logger.info(
            f"Use model.to(dtype) to quantize the model {model.__class__.__name__} to built-in dtype {q_config}"
        )
        return model

    # emulated quantization methods
    quantized_layers = []
    ignored_layers = []

    if isinstance(q_config, str):
        if q_config == "bypass":
            logger.warning("Quantizing model with bypass quantizer, which means no quantization will be applied.")
        q_config = build_default_q_config(q_config)

    # quantize torch.matmul/torch.bmm
    if isinstance(model, PreTrainedModel):
        assert model.config._attn_implementation == "eager", "Only support eager attention implementation"
        quantized_layers += replace_attn_layers(model, q_config["matmul"])

    for name, ori_layer in model.named_modules():
        matched = find_matched_pattern(name, layers_to_ignore)
        if matched is not None:
            # supports skipping layers like lm_head and cls_head
            ignored_layers.append(name)
            continue
        if not isinstance(ori_layer, (torch.nn.Conv2d, torch.nn.Linear)):
            # for now only conv2d and linear are supported
            continue
        if isinstance(ori_layer, torch.nn.Conv2d):
            new_layer = QConv2d(
                ori_layer.in_channels,
                ori_layer.out_channels,
                ori_layer.kernel_size,
                ori_layer.stride,
                ori_layer.padding,
                ori_layer.dilation,
                ori_layer.groups,
                ori_layer.bias is not None,
                ori_layer.padding_mode,
                q_config=q_config["conv2d"] if q_config is not None else None,
            )
        elif isinstance(ori_layer, torch.nn.Linear):
            new_layer = QLinear(
                ori_layer.in_features,
                ori_layer.out_features,
                ori_layer.bias is not None,
                q_config=q_config["linear"] if q_config is not None else None,
            )
        else:
            raise RuntimeError(f"Unsupported layer type: {type(ori_layer)}")

        new_layer.load_state_dict(ori_layer.state_dict())
        new_layer.to(ori_layer.weight.device)
        set_layer_by_name(model, name, new_layer)

        quantized_layers.append(ori_layer.__class__)

    quantized_layers = Counter([layer.__name__ for layer in quantized_layers])
    logger.debug(
        f"Quantized layers ({quantized_layers.total()} in total) in {model.__class__.__name__}:{pformat(dict(quantized_layers), sort_dicts=False)}"
    )

    return model
