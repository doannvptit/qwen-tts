import torch
from torch import nn
from ..modules.res_conv import ResConvBlock


class MelEncoder(nn.Module):
    def __init__(
        self,
        mel_bins: int,
        hidden_size: int,
        num_layers: int,
        kernel_size: int,
        dropout_rate: float,
    ):
        super(MelEncoder, self).__init__()

        self.mel_pre = torch.nn.Sequential(
            torch.nn.Linear(mel_bins, hidden_size),
            nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Dropout(dropout_rate),
        )
        self.mel_encoder = ResConvBlock(
            layers=num_layers,
            n_channels=hidden_size,
            k_size=kernel_size,
            dropout=dropout_rate,
        )

    def forward(self, mels: torch.Tensor):
        """
        mels: Tensor<B, S, D>
        return: Tensor<B, S, D>
        """
        x = self.mel_pre(mels)
        return self.mel_encoder(x)
