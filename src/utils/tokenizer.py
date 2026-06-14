import torch


CHAR_VOCAB = (
    " "
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "àáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵ"
    "đ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
    "ÈÉẺẼẸÊỀẾỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰ"
    "ỲÝỶỸỴ"
    "Đ"
    ".,!?;:-_'\"()/[]{}@#$%&*+=<>|\\\n\t"
)
CHAR_PAD_ID = 0
CHAR_UNK_ID = 1
CHAR_TO_ID = {ch: idx + 2 for idx, ch in enumerate(CHAR_VOCAB)}
CHAR_VOCAB_SIZE = len(CHAR_TO_ID) + 2


def expand_token_embeds_to_chars(
    tokenizer,
    token_ids: list[int],
    token_embeds: torch.Tensor,
) -> torch.Tensor:
    sample_embeds = []

    for token_id, embed in zip(token_ids, token_embeds):
        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        repeat_count = max(len(token_text), 1)
        sample_embeds.append(embed.unsqueeze(0).repeat(repeat_count, 1))

    return torch.cat(sample_embeds, dim=0)


def tokenize_mask_assistant(tokenizer, messages_batch):
    """
    Tokenize and mask assistant messages in a batch of conversations.

    Args:
        tokenizer: Tokenizer instance
        messages_batch: List of conversation histories

    Returns:
        Tuple of (input_ids, attention_mask, assistant_mask)
    """
    encoded = tokenizer.apply_chat_template(
        messages_batch,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        return_assistant_tokens_mask=True,
    )

    assistant_mask = encoded.get("assistant_masks")
    if assistant_mask is None:
        assistant_mask = encoded.get("assistant_mask")
    if assistant_mask is None:
        raise ValueError("assistant mask not returned")

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    assistant_mask = torch.as_tensor(assistant_mask, dtype=torch.bool)
    if assistant_mask.shape != input_ids.shape:
        raise ValueError("assistant mask shape mismatch")
    return input_ids, attention_mask, assistant_mask


def build_assistant_labels(
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    assistant_mask = torch.as_tensor(
        assistant_mask,
        dtype=torch.bool,
        device=input_ids.device,
    )
    if assistant_mask.shape != input_ids.shape:
        raise ValueError("assistant mask shape mismatch")

    labels = input_ids.clone()
    labels.masked_fill_(~assistant_mask, ignore_index)
    return labels


def extract_last_assistant_embeds(hidden_states, assistant_mask):
    """
    Extract last assistant embeds from hidden states.

    Args:
        hidden_states: Hidden states from model
        assistant_mask: Assistant mask

    Returns:
        Last assistant embeds, embeds length
    """
    assistant_mask = assistant_mask.to(dtype=torch.bool, device=hidden_states.device)

    batch_size, seq_len, hidden_dim = hidden_states.shape
    last_spans = []
    span_lengths = []

    for batch_idx in range(batch_size):
        mask_row = assistant_mask[batch_idx]
        indices = torch.nonzero(mask_row, as_tuple=False).squeeze(-1)

        if indices.numel() == 0:
            raise ValueError(
                f"No assistant tokens found for sample {batch_idx}. "
                "Ensure each conversation contains at least one assistant message."
            )

        end_idx = indices[-1].item()
        start_idx = end_idx
        while start_idx > 0 and mask_row[start_idx - 1]:
            start_idx -= 1

        span = hidden_states[batch_idx, start_idx : end_idx + 1]
        last_spans.append(span)
        span_lengths.append(span.shape[0])

    max_len = max(span_lengths)
    output = hidden_states.new_zeros((batch_size, max_len, hidden_dim))

    for batch_idx, span in enumerate(last_spans):
        output[batch_idx, : span.shape[0]] = span

    lengths = torch.tensor(span_lengths, dtype=torch.long, device=hidden_states.device)
    return output, lengths


def extract_last_assistant_char_aligned_embeds(
    tokenizer,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    input_embeds: torch.Tensor,
    assistant_mask: torch.Tensor,
):
    assistant_mask = assistant_mask.to(dtype=torch.bool, device=hidden_states.device)

    if input_embeds.shape != hidden_states.shape:
        raise ValueError("input embeds shape must match hidden states shape")

    batch_size, _, hidden_dim = hidden_states.shape
    expanded_hidden_states = []
    expanded_input_embeds = []
    span_lengths = []

    for batch_idx in range(batch_size):
        mask_row = assistant_mask[batch_idx]
        indices = torch.nonzero(mask_row, as_tuple=False).squeeze(-1)

        if indices.numel() == 0:
            raise ValueError(
                f"No assistant tokens found for sample {batch_idx}. "
                "Ensure each conversation contains at least one assistant message."
            )

        end_idx = indices[-1].item()
        start_idx = end_idx
        while start_idx > 0 and mask_row[start_idx - 1]:
            start_idx -= 1

        span_ids = input_ids[batch_idx, start_idx : end_idx + 1]
        span_hidden = hidden_states[batch_idx, start_idx : end_idx + 1]
        span_input_embeds = input_embeds[batch_idx, start_idx : end_idx + 1]

        sample_hidden = expand_token_embeds_to_chars(
            tokenizer,
            span_ids.tolist(),
            span_hidden,
        )
        sample_input_embeds = expand_token_embeds_to_chars(
            tokenizer,
            span_ids.tolist(),
            span_input_embeds,
        )

        expanded_hidden_states.append(sample_hidden)
        expanded_input_embeds.append(sample_input_embeds)
        span_lengths.append(sample_hidden.size(0))

    max_len = max(span_lengths)
    output_hidden_states = hidden_states.new_zeros((batch_size, max_len, hidden_dim))
    output_input_embeds = input_embeds.new_zeros((batch_size, max_len, hidden_dim))

    for batch_idx, (sample_hidden, sample_input_embed) in enumerate(
        zip(expanded_hidden_states, expanded_input_embeds)
    ):
        sample_len = sample_hidden.size(0)
        output_hidden_states[batch_idx, :sample_len] = sample_hidden
        output_input_embeds[batch_idx, :sample_len] = sample_input_embed

    lengths = torch.tensor(span_lengths, dtype=torch.long, device=hidden_states.device)
    return output_hidden_states, lengths, output_input_embeds
