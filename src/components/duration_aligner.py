import torch
import numpy as np
from torch import nn

from ..utils.tools import get_mask_from_lengths


def pad2ds(x: torch.Tensor, max_t_len: int, max_m_len: int):
    t_len, m_len = x.shape
    return torch.nn.functional.pad(
        x, (0, max_m_len - m_len, 0, max_t_len - t_len), mode="constant", value=1
    )


def generate_W(T1: int, T2: int, g: float = 0.2):
    """

    :param T1: number of text tokens
    :param T2: number of mels
    :param g:
    :return:
    """
    n_items = torch.arange(0, T1) / (T1 - 1)
    t_items = torch.arange(0, T2) / (T2 - 1)
    w = torch.exp(-((n_items.unsqueeze(1) - t_items.unsqueeze(0)) ** 2) / (2 * g**2))
    return w


def generate_weights(embeds_len: list[int], mel_lens: list[int], device):
    max_text_len = max(embeds_len)
    max_mel_len = max(mel_lens)

    e_weight = [
        pad2ds(generate_W(embeds_len[i], mel_lens[i]), max_text_len, max_mel_len)
        for i in range(len(embeds_len))
    ]
    e_weight = torch.stack(e_weight).to(device)
    return e_weight


def scaled_dot_attention(key, key_lens, query, query_lens, e_weight=None):
    dim = key.size(-1)
    T1 = query.size(1)
    N1 = key.size(1)
    device = key.device
    energies = query @ key.transpose(1, 2) / np.sqrt(float(dim))
    if e_weight is not None:
        energies = energies * e_weight.transpose(1, 2)

    key_mask = get_mask_from_lengths(key_lens, max_len=N1).to(device)
    key_mask = key_mask.unsqueeze(1).repeat(1, T1, 1)
    energies = energies.masked_fill(key_mask, -float("inf"))
    alpha = torch.softmax(energies, dim=-1)

    query_mask = get_mask_from_lengths(query_lens, max_len=T1).to(device)
    query_mask = query_mask.unsqueeze(2).repeat(1, 1, N1)
    alpha = alpha.masked_fill(query_mask, 0.0)
    return alpha.transpose(1, 2)


class DurationAligner(nn.Module):
    def __init__(self):
        super(DurationAligner, self).__init__()

    def forward(
        self,
        embeds: torch.Tensor,
        embeds_len: torch.Tensor,
        mel_h: torch.Tensor,
        mel_lens: torch.Tensor,
    ):
        """
        embeds: Tensor<B, T1, IN>
        embeds_len: Tensor<B>
        mel_h: Tensor<B, T2, D>
        mel_lens: Tensor<B>
        """
        embeds_len_list = embeds_len.tolist()
        mel_lens_list = mel_lens.tolist()
        e_weight = generate_weights(
            embeds_len_list, mel_lens_list, device=embeds.device
        )

        alpha = scaled_dot_attention(
            key=embeds,
            key_lens=embeds_len,
            query=mel_h,
            query_lens=mel_lens,
            e_weight=e_weight,
        )
        return torch.sum(alpha, dim=-1)
