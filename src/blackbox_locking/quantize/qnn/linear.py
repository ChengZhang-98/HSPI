from copy import deepcopy
from functools import partial

import torch.nn as nn
from torch.nn import functional as F

from ..quantizers import get_quantizer


class QLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        q_config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        if q_config is None:
            self.bypass = True
        else:
            self.bypass = q_config.get("bypass", False)

        self.q_config = q_config

        if not self.bypass:
            x_q_config = deepcopy(q_config["x"])
            w_q_config = deepcopy(q_config["w"])

            self.x_quantizer = partial(get_quantizer(x_q_config.pop("name")), **x_q_config)
            self.w_quantizer = partial(get_quantizer(w_q_config.pop("name")), **w_q_config)

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        else:
            x = self.x_quantizer(x)
            w = self.w_quantizer(self.weight)

            bias = None
            if self.bias is not None:
                bias = self.w_quantizer(self.bias)

            return F.linear(x, w, bias)

    def __repr__(self):
        if self.bypass:
            q_config_str = "bypass=True"
        else:
            q_config_str = f"x_quantizer={self.q_config['x']['name']}, w_quantizer={self.q_config['w']['name']}"
        return f"QLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, {q_config_str})"
