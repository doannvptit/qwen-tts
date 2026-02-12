import torch


def get_mask_from_lengths(lengths, max_len=None):
    """
    :param lengths: [B]
    :param max_len:
    :return:
        [B, max_len] where True is padding and False is valid
    """
    batch_size = lengths.shape[0]
    device = lengths.device
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask
