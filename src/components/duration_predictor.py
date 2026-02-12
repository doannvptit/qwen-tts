import torch
from torch import nn
from ..modules.conv import Conv1d


class DurationPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        k_size: int,
        layers: int,
        dropout_rate: float,
    ):
        super(DurationPredictor, self).__init__()

        self.pre_block = nn.Sequential(
            Conv1d(
                in_channels=input_size,
                out_channels=hidden_size,
                kernel_size=k_size,
                padding=(k_size - 1) // 2,
                auto_transpose=True,
                causal=True,
            ),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1d(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size=k_size,
                        padding=(k_size - 1) // 2,
                        auto_transpose=True,
                        causal=True,
                    ),
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(dropout_rate),
                )
                for _ in range(layers - 1)
            ]
        )
        self.projection = nn.Linear(hidden_size, 1)
        self.offset = 1

    def forward(self, x, is_train=True):
        """
        x: Tensor<B, T, D>
        is_train: bool
        return: Tensor<B, T>
        """

        x = self.pre_block(x)
        for layer in self.layers:
            x = layer(x)
        x = self.projection(x)
        x = x.squeeze(-1)
        x = x.exp() - self.offset
        if not is_train:
            x = torch.clamp(x, min=0)
        return x
