from .integer import int_quantizer
from .minifloat import minifloat_ieee_quantizer
from .mxint import mxint_quantizer
from .block_minifloat import block_minifloat_quantizer
from .log import log_quantizer
from .block_log import block_log_quantizer
from .torch_built_in import bf16_quantizer, fp16_quantizer, fp32_quantizer
from .bypass import bypass_quantizer


def get_quantizer(name: str):
    match name:
        case "integer":
            return int_quantizer
        case "minifloat":
            return minifloat_ieee_quantizer
        case "block_minifloat":
            return block_minifloat_quantizer
        case "logarithm":
            return log_quantizer
        case "block_logarithm":
            return block_log_quantizer
        case "bypass":
            return bypass_quantizer
        case "mxint":
            return mxint_quantizer
        case "bf16":
            return bf16_quantizer
        case "fp16":
            return fp16_quantizer
        case "fp32":
            return fp32_quantizer
        case _:
            raise ValueError(f"Unknown quantizer name {name}")
