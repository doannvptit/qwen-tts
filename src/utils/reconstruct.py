import torch
from .tools import get_mask_from_lengths


def build_batch_selection_ids(e, mel_lens):
    e = e.long().cpu().tolist()
    batch_selection_ids = []
    for i, duration in enumerate(e):
        selection_ids = []
        for j in range(len(duration)):
            extend_size = duration[j] - len(selection_ids)
            if extend_size > 0:
                selection_ids.extend([j] * extend_size)

        if mel_lens is not None:
            if mel_lens[i] > len(selection_ids):
                selection_ids.extend(
                    [selection_ids[-1]] * (mel_lens[i] - len(selection_ids))
                )
            elif mel_lens[i] < len(selection_ids):
                selection_ids = selection_ids[: mel_lens[i]]

        batch_selection_ids.append(selection_ids)
    return batch_selection_ids


def build_causal_select_mask(duration_target, max_mel_len, look_ahead):
    device = duration_target.device
    duration_target = torch.cumsum(duration_target, dim=-1).ceil().long()

    causal_select_masks = []
    for duration in duration_target:
        # shift to right and push 0 at first
        shift = look_ahead + 1
        duration = torch.cat(
            [torch.zeros(shift).long().to(device), duration[:-shift]], dim=0
        )
        sub_causal = ~get_mask_from_lengths(duration, max_len=max_mel_len).to(device)
        causal_select_masks.append(sub_causal)
    return torch.stack(causal_select_masks)


def reconstruct_align_from_aligned_position(
    duration_target,
    delta=0.2,
    mel_lens=None,
    embeds_len=None,
    max_mel_len=None,
    look_ahead=10,
):
    """Reconstruct alignment matrix from aligned positions.
    Args:
        e: aligned positions [B, T1].
        delta: a scalar, default 0.2
        mel_mask: mask of mel-spectrogram [B, T2], None if inference and B==1.
        text_mask: mask of text-sequence, None if B==1.
    Returns:
        alignment matrix [B, T1, T2].
    """
    e = torch.cumsum(duration_target, dim=-1)
    e = e - duration_target / 2
    b, T1 = e.shape
    if mel_lens is None:
        mel_lens = torch.round(e[:, -1]).long()

    if max_mel_len is None:
        max_length = mel_lens.max().item()
    else:
        max_length = max_mel_len

    q = (
        torch.arange(0, max_length)
        .unsqueeze(0)
        .repeat(e.size(0), 1)
        .to(e.device)
        .float()
    )

    mel_mask = get_mask_from_lengths(mel_lens, max_len=max_length).to(e.device)
    q = q * (~mel_mask).float()

    energies = -1 * delta * (q.unsqueeze(1) - e.unsqueeze(-1)) ** 2
    if embeds_len is not None:
        text_mask = get_mask_from_lengths(embeds_len, max_len=T1).to(e.device)
        text_mask = text_mask.unsqueeze(-1).repeat(1, 1, max_length)
        energies = energies.masked_fill(text_mask, -float("inf"))

    causal_select_mask = build_causal_select_mask(
        duration_target, energies.shape[-1], look_ahead
    )
    energies = energies.masked_fill(causal_select_mask, -float("inf"))

    alpha = torch.softmax(energies, dim=1)
    alpha = alpha.masked_fill(
        mel_mask.unsqueeze(1).repeat(1, text_mask.size(1), 1), 0.0
    )

    return alpha, mel_lens
