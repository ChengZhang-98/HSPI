import logging
import torch

logger = logging.getLogger(__name__)

def bf16_quantizer(x: torch.Tensor):
    ori_dtype = x.dtype
    x = x.to(torch.bfloat16).to(ori_dtype)
    return x

def fp16_quantizer(x: torch.Tensor):
    ori_dtype = x.dtype
    x = x.to(torch.float16).to(ori_dtype)
    return x

def fp32_quantizer(x: torch.Tensor):
    ori_dtype = x.dtype
    x = x.to(torch.float32).to(ori_dtype)
    return x