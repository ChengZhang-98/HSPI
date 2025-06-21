import logging
from pprint import pformat

import torch
from torchvision.models import get_model

from blackbox_locking.quantize import quantize_model
from blackbox_locking.utils import set_seed

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

set_seed(0)


@torch.no_grad()
def test_quantization_bypass():
    # print(list_models())
    model_ref = get_model("resnet18", weights="DEFAULT").eval().cuda()
    model_q = get_model("resnet18", weights="DEFAULT").eval().cuda()
    quantize_model(model_q, "bypass")
    logger.debug(model_q)

    x = torch.randn(1, 3, 224, 224).cuda()

    y_ref = model_ref(x)
    y_q = model_q(x)

    assert torch.all(y_ref.eq(y_q)), "Quantize bypass failed"


@torch.no_grad()
def test_quantization_error():
    """
    {'fp32': 0.0,
    'fp16': 0.0010209056781604886,
    'bf16': 0.013747152872383595,
    'bypass': 0.0,
    'int8-dynamic': 0.14469973742961884,
    'fp8-e4m3': 0.10163913667201996,
    'fp8-e3m4': 0.09834522753953934,
    'mxint8': 0.037580110132694244,
    'bm8': 2.792686700820923,
    'bl8': 1.582614779472351,
    'log8': 1.7531641721725464}
    """
    model_ref = get_model("resnet18", weights="DEFAULT").eval().cuda()
    name_error = {}
    for q_name in [
        # built-in:
        "fp32",
        "fp16",
        "bf16",
        "bypass",
        # emulated:
        ## relatively precise
        "int8-dynamic",
        "fp8-e4m3",
        "fp8-e3m4",
        "mxint8",
        ## may cause significant error
        "bm8",
        "bl8",
        "log8",
    ]:
        model_q = get_model("resnet18", weights="DEFAULT").eval().cuda()
        quantize_model(model_q, q_name)
        x = torch.randn(1, 3, 224, 224, device="cuda")
        y_ref = model_ref(x)
        y_q = model_q(x)

        error = (y_ref - y_q).abs().mean().cpu().item()
        name_error[q_name] = error

    logger.debug(f"Model output quantization error:\n{pformat(name_error, sort_dicts=False)}")


if __name__ == "__main__":
    # test_quantization_bypass()
    test_quantization_error()
