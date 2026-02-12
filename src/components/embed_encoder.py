import torch
import torch.nn as nn

from ..modules.transformer import EncoderLayers
from ..utils.tools import get_mask_from_lengths


class EmbedEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        layers: int,
        heads: int,
        hidden: int,
        conv_kernel_size: list[int],
        dropout: float,
        max_position: int,
        output_size: int,
    ):
        super(EmbedEncoder, self).__init__()
        self.pre_project = nn.Linear(input_size, hidden)
        self.encoder = EncoderLayers(
            num_layers=layers,
            d_model=hidden,
            ffn_hidden=hidden * 4,
            n_head=heads,
            kernel_size=conv_kernel_size,
            drop_prob=dropout,
            causal=True,
            max_position=max_position,
        )
        self.project_key = nn.Linear(hidden, output_size)
        self.project_value = nn.Linear(hidden, output_size)

    def forward(
        self, embeds: torch.Tensor, embeds_lens: torch.Tensor
    ):
        """
        embeds: [B, T, D]
        embeds_lens: [B]
        """
        embedded = self.pre_project(embeds)

        src_mask = get_mask_from_lengths(embeds_lens)
        x = self.encoder(embedded, src_mask)
        return self.project_key(x), self.project_value(x)
