import unittest

import torch
from torch import nn

from src.talker import Talker, TalkerConfig


class CaptureEmbedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, embeds, embeds_len):
        self.last_input = embeds.detach().clone()
        return embeds, embeds


class DummyDurationAligner(nn.Module):
    def forward(self, embeds, embeds_len, mel_h, mels_len):
        return torch.zeros(
            embeds.size(0),
            embeds.size(1),
            dtype=embeds.dtype,
            device=embeds.device,
        )


class DummyDurationPredictor(nn.Module):
    def forward(self, embeds_value):
        return torch.zeros(
            embeds_value.size(0),
            embeds_value.size(1),
            dtype=embeds_value.dtype,
            device=embeds_value.device,
        )


class DummyMelEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, mels):
        return torch.zeros(
            mels.size(0),
            mels.size(1),
            self.hidden_dim,
            dtype=mels.dtype,
            device=mels.device,
        )


class DummyMelDecoder(nn.Module):
    def __init__(self, mel_bins):
        super().__init__()
        self.mel_bins = mel_bins

    def forward(self, embed_expanded):
        return torch.zeros(
            embed_expanded.size(0),
            embed_expanded.size(1),
            self.mel_bins,
            dtype=embed_expanded.dtype,
            device=embed_expanded.device,
        )


class IdentityPostNet(nn.Module):
    def forward(self, mel_pred):
        return mel_pred


class TestTalkerVocabEmbedding(unittest.TestCase):
    def test_talker_adds_projected_vocab_embedding_before_text_encoder(self):
        cfg = TalkerConfig(
            input_dim=4,
            hidden_dim=4,
            encoder_layers=1,
            encoder_heads=1,
            encoder_conv_kernel_size=[3, 1],
            encoder_max_position=32,
            duration_predictor_kernel_size=3,
            duration_predictor_num_layers=1,
            mel_encoder_kernel_size=3,
            mel_encoder_num_layers=1,
            mel_decoder_kernel_size=3,
            mel_decoder_num_layers=1,
            post_net_kernel_size=3,
            post_net_num_layers=1,
            dropout_rate=0.0,
            mel_bins=5,
            delta=0.2,
            look_ahead=2,
            char_vocab_size=16,
        )
        talker = Talker(cfg, device=torch.device("cpu"), dtype=torch.float32)

        vocab_embed_mlp = nn.Linear(cfg.input_dim, cfg.input_dim, bias=False)
        with torch.no_grad():
            vocab_embed_mlp.weight.copy_(2.0 * torch.eye(cfg.input_dim))
        talker.vocab_embed_mlp = vocab_embed_mlp

        capture_encoder = CaptureEmbedEncoder()
        talker.embed_encoder = capture_encoder
        talker.duration_aligner = DummyDurationAligner()
        talker.duration_predictor = DummyDurationPredictor()
        talker.mel_encoder = DummyMelEncoder(hidden_dim=cfg.hidden_dim)
        talker.mel_decoder = DummyMelDecoder(mel_bins=cfg.mel_bins)
        talker.post_net = IdentityPostNet()
        talker._expand_duration = (
            lambda duration_target, embeds_value, embeds_len, mel_lens: (
                embeds_value,
                mel_lens,
            )
        )

        embeds = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5], [2.0, 1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )
        vocab_embeds = torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4], [1.0, 1.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.5]]],
            dtype=torch.float32,
        )
        embeds_len = torch.tensor([3], dtype=torch.long)

        mels = torch.zeros((1, 3, cfg.mel_bins), dtype=torch.float32)
        mels_len = torch.tensor([3], dtype=torch.long)

        output = talker(embeds, embeds_len, vocab_embeds, mels, mels_len)

        expected_encoder_input = embeds + (2.0 * vocab_embeds)
        self.assertTrue(
            torch.allclose(capture_encoder.last_input, expected_encoder_input)
        )
        self.assertIn("mel_pred", output)
        self.assertIn("mel_post", output)
        self.assertEqual(tuple(output["mel_pred"].shape), (1, 3, cfg.mel_bins))

    def test_vocab_embedding_mlp_trains_without_updating_vocab_embeds(self):
        cfg = TalkerConfig(
            input_dim=4,
            hidden_dim=4,
            encoder_layers=1,
            encoder_heads=1,
            encoder_conv_kernel_size=[3, 1],
            encoder_max_position=32,
            duration_predictor_kernel_size=3,
            duration_predictor_num_layers=1,
            mel_encoder_kernel_size=3,
            mel_encoder_num_layers=1,
            mel_decoder_kernel_size=3,
            mel_decoder_num_layers=1,
            post_net_kernel_size=3,
            post_net_num_layers=1,
            dropout_rate=0.0,
            mel_bins=5,
            delta=0.2,
            look_ahead=2,
            char_vocab_size=16,
        )
        talker = Talker(cfg, device=torch.device("cpu"), dtype=torch.float32)

        embeds = torch.zeros((1, 2, cfg.input_dim), dtype=torch.float32)
        vocab_embeds = torch.ones(
            (1, 2, cfg.input_dim), dtype=torch.float32, requires_grad=True
        )

        loss = talker._prepare_embeds(embeds, vocab_embeds).sum()
        loss.backward()

        self.assertIsNone(vocab_embeds.grad)
        mlp_grads = [
            param.grad
            for param in talker.vocab_embed_mlp.parameters()
            if param.requires_grad
        ]
        self.assertTrue(any(grad is not None for grad in mlp_grads))


if __name__ == "__main__":
    unittest.main()
