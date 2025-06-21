import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.phi3.modeling_phi3 import Phi3Attention
from .llama import create_q_llama_attn, quantize_llama_layer_to_built_in_dtype
from .phi3 import create_q_phi3_attn, quantize_phi3_layer_to_built_in_dtype
from ...utils import set_layer_by_name


def quantize_hf_to_builtin_dtype(model: PreTrainedModel, dtype: torch.dtype):
    """
    HuggingFace PreTrainedModel may need special handling for quantization such as positional encoding.
    """
    match model.__class__.__name__:
        case "LlamaForCausalLM":
            return quantize_llama_layer_to_built_in_dtype(model, dtype)
        case "Phi3ForCausalLM":
            return quantize_phi3_layer_to_built_in_dtype(model, dtype)
        case _:
            raise ValueError(f"Quantization not supported for {model.__class__.__name__}")


def get_ori_attention_cls(model: PreTrainedModel):
    match model.__class__.__name__:
        case "LlamaForCausalLM":
            return LlamaAttention
        case "Phi3ForCausalLM":
            return Phi3Attention
        case _:
            raise ValueError(f"Quantization not supported for {model.__class__.__name__}")


def create_q_attention(ori_attention, q_config):
    match ori_attention.__class__.__name__:
        case "LlamaAttention":
            return create_q_llama_attn(ori_attention, q_config)
        case "Phi3Attention":
            return create_q_phi3_attn(ori_attention, q_config)
        case _:
            raise ValueError(f"Quantization not supported for {ori_attention.__class__.__name__}")


def replace_attn_layers(model: torch.nn.Module, q_config: dict):
    """
    Replace the attention layers such that the two attention Matmul operations are quantized.
    """
    ori_attn_cls = get_ori_attention_cls(model)
    replaced_layers = []
    for name, ori_m in model.named_modules():
        if not isinstance(ori_m, ori_attn_cls):
            continue

        new_m = create_q_attention(ori_m, q_config)
        set_layer_by_name(model, name, new_m)

        replaced_layers.append(ori_attn_cls)

    return replaced_layers
