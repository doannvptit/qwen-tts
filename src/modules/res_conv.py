import torch
from torch import nn

from .conv import Conv1d


class ResConv1D(nn.Module):
    def __init__(self, n_channels, k_size, dropout=0.1, causal=False):
        super(ResConv1D, self).__init__()
        self.conv = torch.nn.Sequential(
            Conv1d(
                n_channels,
                n_channels,
                kernel_size=k_size,
                padding=(k_size - 1) // 2,
                causal=causal,
                auto_transpose=True,
            ),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.conv(x)
        return x


class ResConvBlock(nn.Module):
    def __init__(
        self,
        layers,
        n_channels,
        k_size,
        dropout=0.1,
        causal=False,
    ):
        super(ResConvBlock, self).__init__()
        self.layers = nn.ModuleList(
            [ResConv1D(n_channels, k_size, dropout, causal) for _ in range(layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
