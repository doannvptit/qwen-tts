import torch.nn as nn
import torch

"""
Causal Convolution Module by force set right side of weight to zero
"""


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        auto_transpose=False,
        causal=False,
    ):
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        self.auto_transpose = auto_transpose
        if causal:
            with torch.no_grad():
                # create causal mask
                mask = torch.ones_like(self.weight.data)
                mask[:, :, (self.kernel_size[0] // 2 + 1) :] = 0
                self.register_buffer("mask", mask)
                self.weight *= self.mask

            # register hook to keep mask after each backward
            def apply_mask(grad):
                return grad * self.mask

            self.weight.register_hook(apply_mask)

    def forward(self, x):
        if self.auto_transpose:
            x = x.contiguous().transpose(1, 2)
        x = super().forward(x)
        if self.auto_transpose:
            x = x.contiguous().transpose(1, 2)

        return x


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        auto_transpose=False,
        causal=False,
    ):
        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )

        self.auto_transpose = auto_transpose

        if causal:
            with torch.no_grad():
                # build causal mask
                mask = torch.ones_like(self.weight.data)
                # weight shape: (in_channels, out_channels, k)
                # zero future (right-side) weights
                mask[..., (self.kernel_size[0] // 2 + 1) :] = 0
                self.register_buffer("mask", mask)
                self.weight *= self.mask

            # register hook so masked weights remain zero after backward
            def apply_mask(grad):
                return grad * self.mask

            self.weight.register_hook(apply_mask)

    def forward(self, x):
        if self.auto_transpose:
            x = x.transpose(1, 2).contiguous()
        x = super().forward(x)
        if self.auto_transpose:
            x = x.transpose(1, 2).contiguous()
        return x
