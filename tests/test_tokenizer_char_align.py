import unittest

import torch

from src.utils.tokenizer import (
    CHAR_TO_ID,
    CHAR_UNK_ID,
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
    def test_repeat_embed_by_decoded_char_count_and_map_char_ids(self):
        tokenizer = DummyTokenizer({10: "xin", 11: " chào"})

        input_ids = torch.tensor([[1, 2, 10, 11]], dtype=torch.long)
        assistant_mask = torch.tensor([[False, False, True, True]], dtype=torch.bool)
        hidden_states = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [1.0, 1.5], [2.0, 2.5]]], dtype=torch.float32
        )

        embeds, lengths, char_ids = extract_last_assistant_char_aligned_embeds(
            tokenizer=tokenizer,
            input_ids=input_ids,
            hidden_states=hidden_states,
            assistant_mask=assistant_mask,
        )

        self.assertEqual(lengths.tolist(), [8])
        self.assertEqual(tuple(embeds.shape), (1, 8, 2))
        self.assertEqual(tuple(char_ids.shape), (1, 8))

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

        expected_chars = [
            CHAR_TO_ID["x"],
            CHAR_TO_ID["i"],
            CHAR_TO_ID["n"],
            CHAR_TO_ID[" "],
            CHAR_TO_ID["c"],
            CHAR_TO_ID["h"],
            CHAR_TO_ID["à"],
            CHAR_TO_ID["o"],
        ]
        self.assertEqual(char_ids[0].tolist(), expected_chars)

    def test_empty_decoded_token_maps_to_unk_and_single_repeat(self):
        tokenizer = DummyTokenizer({7: ""})

        input_ids = torch.tensor([[7]], dtype=torch.long)
        assistant_mask = torch.tensor([[True]], dtype=torch.bool)
        hidden_states = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32)

        embeds, lengths, char_ids = extract_last_assistant_char_aligned_embeds(
            tokenizer=tokenizer,
            input_ids=input_ids,
            hidden_states=hidden_states,
            assistant_mask=assistant_mask,
        )

        self.assertEqual(lengths.tolist(), [1])
        self.assertTrue(torch.equal(embeds[0, 0], torch.tensor([3.0, 4.0])))
        self.assertEqual(char_ids[0, 0].item(), CHAR_UNK_ID)

    def test_unknown_character_in_decoded_text_maps_to_unk_id(self):
        tokenizer = DummyTokenizer({42: "a🙂"})

        input_ids = torch.tensor([[42]], dtype=torch.long)
        assistant_mask = torch.tensor([[True]], dtype=torch.bool)
        hidden_states = torch.tensor([[[5.0, 6.0]]], dtype=torch.float32)

        embeds, lengths, char_ids = extract_last_assistant_char_aligned_embeds(
            tokenizer=tokenizer,
            input_ids=input_ids,
            hidden_states=hidden_states,
            assistant_mask=assistant_mask,
        )

        self.assertEqual(lengths.tolist(), [2])
        self.assertEqual(tuple(embeds.shape), (1, 2, 2))
        self.assertTrue(torch.equal(embeds[0, 0], hidden_states[0, 0]))
        self.assertTrue(torch.equal(embeds[0, 1], hidden_states[0, 0]))

        self.assertEqual(char_ids[0, 0].item(), CHAR_TO_ID["a"])
        self.assertEqual(char_ids[0, 1].item(), CHAR_UNK_ID)


if __name__ == "__main__":
    unittest.main()
