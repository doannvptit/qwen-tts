import torch


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
