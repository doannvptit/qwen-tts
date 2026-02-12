from torch import nn
from ..modules.res_conv import ResConvBlock


class MelDecoder(nn.Module):
    def __init__(
        self,
        mel_bins: int,
        hidden_size: int,
        num_layers: int,
        kernel_size: int,
        dropout_rate: float,
    ):
        super(MelDecoder, self).__init__()

        self.decoder = ResConvBlock(
            layers=num_layers,
            n_channels=hidden_size,
            k_size=kernel_size,
            dropout=dropout_rate,
            causal=True,
        )
        self.projection = nn.Linear(hidden_size, mel_bins)

    def forward(self, x):
        """
        x: Tensor<B, S, D>
        return: Tensor<B, S, mel_bins>
        """
        x = self.decoder(x)
        return self.projection(x)
