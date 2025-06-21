from copy import deepcopy
from functools import partial

import torch.nn as nn
from torch.nn import functional as F

from ..quantizers import get_quantizer


class QConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | F.Tuple[int],
        stride: int | F.Tuple[int] = 1,
        padding: str | int | F.Tuple[int] = 0,
        dilation: int | F.Tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        q_config=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

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
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            x = self.x_quantizer(x)
            try:
                w = self.w_quantizer(self.weight)
            except Exception as e:
                breakpoint()
                print(e)

            bias = None
            if self.bias is not None:
                bias = self.w_quantizer(self.bias)

            return F.conv2d(
                x,
                w,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def __repr__(self):
        if self.bypass:
            q_config_str = "bypass=True"
        else:
            q_config_str = f"x_quantizer={self.q_config['x']['name']}, w_quantizer={self.q_config['w']['name']}"

        return f"QConv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}, padding_mode={self.padding_mode}, {q_config_str})"
