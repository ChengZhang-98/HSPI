from copy import deepcopy
import torch
from ..quantizers import get_quantizer


def q_matmul(input: torch.Tensor, other: torch.Tensor, q_config: dict):
    if q_config is None:
        return torch.matmul(input, other)
    else:
        x_config = deepcopy(q_config["x"])
        w_config = deepcopy(q_config["w"])

        x_q = get_quantizer(x_config.pop("name"))(input, **x_config)
        w_q = get_quantizer(w_config.pop("name"))(other, **w_config)

        return torch.matmul(x_q, w_q)
