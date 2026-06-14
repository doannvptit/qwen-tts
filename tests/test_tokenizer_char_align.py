import unittest

import torch

from src.utils.tokenizer import (
    build_assistant_labels,
    extract_last_assistant_char_aligned_embeds,
)


class DummyTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def decode(
        self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ):
        return self.mapping.get(token_ids[0], "")


class TestCharAlignedEmbeds(unittest.TestCase):
    def test_build_assistant_labels_masks_non_assistant_tokens(self):
        input_ids = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]], dtype=torch.long)
        assistant_mask = torch.tensor(
            [[False, True, True, False], [False, False, True, False]],
            dtype=torch.bool,
        )

        labels = build_assistant_labels(input_ids, assistant_mask)

        expected = torch.tensor(
            [[-100, 12, 13, -100], [-100, -100, 23, -100]], dtype=torch.long
        )
        self.assertTrue(torch.equal(labels, expected))

    def test_build_assistant_labels_raises_on_shape_mismatch(self):
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        assistant_mask = torch.tensor([[True, False]], dtype=torch.bool)

        with self.assertRaises(ValueError):
            build_assistant_labels(input_ids, assistant_mask)

    def test_repeat_hidden_and_vocab_embeds_by_decoded_char_count(self):
        tokenizer = DummyTokenizer({10: "xin", 11: " chào"})

        input_ids = torch.tensor([[1, 2, 10, 11]], dtype=torch.long)
        assistant_mask = torch.tensor([[False, False, True, True]], dtype=torch.bool)
        hidden_states = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [1.0, 1.5], [2.0, 2.5]]], dtype=torch.float32
        )

        input_embeds = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [10.0, 10.5], [20.0, 20.5]]],
            dtype=torch.float32,
        )

        embeds, lengths, vocab_embeds = extract_last_assistant_char_aligned_embeds(
            tokenizer=tokenizer,
            input_ids=input_ids,
            hidden_states=hidden_states,
            input_embeds=input_embeds,
            assistant_mask=assistant_mask,
        )

        self.assertEqual(lengths.tolist(), [8])
        self.assertEqual(tuple(embeds.shape), (1, 8, 2))
        self.assertEqual(tuple(vocab_embeds.shape), (1, 8, 2))

        expected = torch.tensor(
            [
                [1.0, 1.5],
                [1.0, 1.5],
                [1.0, 1.5],
                [2.0, 2.5],
                [2.0, 2.5],
                [2.0, 2.5],
                [2.0, 2.5],
                [2.0, 2.5],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(embeds[0], expected))

        expected_vocab_embeds = torch.tensor(
            [
                [10.0, 10.5],
                [10.0, 10.5],
                [10.0, 10.5],
                [20.0, 20.5],
                [20.0, 20.5],
                [20.0, 20.5],
                [20.0, 20.5],
                [20.0, 20.5],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(vocab_embeds[0], expected_vocab_embeds))

    def test_empty_decoded_token_uses_single_repeat(self):
        tokenizer = DummyTokenizer({7: ""})

        input_ids = torch.tensor([[7]], dtype=torch.long)
        assistant_mask = torch.tensor([[True]], dtype=torch.bool)
        hidden_states = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32)

        input_embeds = torch.tensor([[[30.0, 40.0]]], dtype=torch.float32)

        embeds, lengths, vocab_embeds = extract_last_assistant_char_aligned_embeds(
            tokenizer=tokenizer,
            input_ids=input_ids,
            hidden_states=hidden_states,
            input_embeds=input_embeds,
            assistant_mask=assistant_mask,
        )

        self.assertEqual(lengths.tolist(), [1])
        self.assertTrue(torch.equal(embeds[0, 0], torch.tensor([3.0, 4.0])))
        self.assertTrue(torch.equal(vocab_embeds[0, 0], torch.tensor([30.0, 40.0])))

    def test_raises_when_input_embed_shape_mismatches_hidden_states(self):
        tokenizer = DummyTokenizer({42: "a"})

        input_ids = torch.tensor([[42]], dtype=torch.long)
        assistant_mask = torch.tensor([[True]], dtype=torch.bool)
        hidden_states = torch.tensor([[[5.0, 6.0]]], dtype=torch.float32)
        input_embeds = torch.tensor([[[5.0, 6.0, 7.0]]], dtype=torch.float32)

        with self.assertRaises(ValueError):
            extract_last_assistant_char_aligned_embeds(
                tokenizer=tokenizer,
                input_ids=input_ids,
                hidden_states=hidden_states,
                input_embeds=input_embeds,
                assistant_mask=assistant_mask,
            )


if __name__ == "__main__":
    unittest.main()
