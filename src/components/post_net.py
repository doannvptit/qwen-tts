import torch.nn as nn

from ..modules.conv import Conv1d


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels: int,
        postnet_embedding_dim: int,
        postnet_kernel_size: int,
        postnet_n_convolutions: int,
    ):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                Conv1d(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    padding=int((postnet_kernel_size - 1) / 2),
                    causal=True,
                    auto_transpose=True,
                ),
                nn.Tanh(),
                nn.Dropout(0.5),
            )
        )

        for i in range(0, postnet_n_convolutions - 2):
            self.convolutions.append(
                nn.Sequential(
                    Conv1d(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        padding=int((postnet_kernel_size - 1) / 2),
                        causal=True,
                        auto_transpose=True,
                    ),
                    nn.Tanh(),
                    nn.Dropout(0.5),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                Conv1d(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    padding=int((postnet_kernel_size - 1) / 2),
                    causal=True,
                    auto_transpose=True,
                ),
                nn.Dropout(0.5),
            )
        )

    def forward(self, x):
        residual = x

        for layer in self.convolutions:
            x = layer(x)

        return x + residual
