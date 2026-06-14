from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .components.embed_encoder import EmbedEncoder
from .components.duration_aligner import DurationAligner
from .components.duration_predictor import DurationPredictor
from .components.mel_decoder import MelDecoder
from .components.mel_encoder import MelEncoder
from .components.post_net import PostNet
from .utils.tools import get_mask_from_lengths
from .utils.reconstruct import reconstruct_align_from_aligned_position


@dataclass
class TalkerConfig:
    input_dim: int
    hidden_dim: int

    encoder_layers: int
    encoder_heads: int
    encoder_conv_kernel_size: list[int]
    encoder_max_position: int

    duration_predictor_kernel_size: int
    duration_predictor_num_layers: int
    mel_encoder_kernel_size: int
    mel_encoder_num_layers: int
    mel_decoder_kernel_size: int
    mel_decoder_num_layers: int
    post_net_kernel_size: int
    post_net_num_layers: int
    dropout_rate: float
    mel_bins: int

    delta: float
    look_ahead: int
    char_vocab_size: int = 512

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class Talker(nn.Module):
    def __init__(self, cfg: TalkerConfig, device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.delta = cfg.delta
        self.look_ahead = cfg.look_ahead

        self.embed_encoder = EmbedEncoder(
            input_size=cfg.input_dim,
            layers=cfg.encoder_layers,
            heads=cfg.encoder_heads,
            hidden=cfg.hidden_dim,
            conv_kernel_size=cfg.encoder_conv_kernel_size,
            dropout=cfg.dropout_rate,
            max_position=cfg.encoder_max_position,
            output_size=cfg.hidden_dim,
        )
        self.duration_predictor = DurationPredictor(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            k_size=cfg.duration_predictor_kernel_size,
            layers=cfg.duration_predictor_num_layers,
            dropout_rate=cfg.dropout_rate,
        )
        self.duration_aligner = DurationAligner()
        self.mel_encoder = MelEncoder(
            mel_bins=cfg.mel_bins,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.mel_encoder_num_layers,
            kernel_size=cfg.mel_encoder_kernel_size,
            dropout_rate=cfg.dropout_rate,
        )
        self.mel_decoder = MelDecoder(
            mel_bins=cfg.mel_bins,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.mel_decoder_num_layers,
            kernel_size=cfg.mel_decoder_kernel_size,
            dropout_rate=cfg.dropout_rate,
        )
        self.post_net = PostNet(
            n_mel_channels=cfg.mel_bins,
            postnet_embedding_dim=cfg.hidden_dim,
            postnet_kernel_size=cfg.post_net_kernel_size,
            postnet_n_convolutions=cfg.post_net_num_layers,
        )
        self.vocab_embed_mlp = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
        )
        self.to(device=device, dtype=dtype)

    def _expand_duration(
        self, duration_target, embeds_value, embeds_len, mel_lens=None
    ):
        reconst_alpha, reconst_mel_lens = reconstruct_align_from_aligned_position(
            duration_target,
            mel_lens=mel_lens,
            embeds_len=embeds_len,
            delta=self.delta,
            look_ahead=self.look_ahead,
        )
        reconst_alpha = reconst_alpha.to(embeds_value.dtype)
        embed_value_expanded = torch.bmm(
            embeds_value.transpose(1, 2), reconst_alpha
        ).transpose(1, 2)
        return embed_value_expanded, reconst_mel_lens

    def _prepare_embeds(
        self,
        embeds: torch.Tensor,
        vocab_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return embeds + self.vocab_embed_mlp(vocab_embeds.detach())

    def forward(
        self,
        embeds: torch.Tensor,
        embeds_len: torch.Tensor,
        vocab_embeds: torch.Tensor,
        mels: torch.Tensor,
        mels_len: torch.Tensor,
    ):
        embeds = self._prepare_embeds(embeds, vocab_embeds)

        mel_mask = get_mask_from_lengths(mels_len, max_len=mels.size(1))
        embeds_key, embeds_value = self.embed_encoder(embeds, embeds_len)
        embeds_mask = get_mask_from_lengths(embeds_len)

        mel_h = self.mel_encoder(mels)
        duration_target = self.duration_aligner(embeds_key, embeds_len, mel_h, mels_len)
        duration_target = duration_target.masked_fill(embeds_mask, 0.0)
        duration_pred = self.duration_predictor(embeds_value)

        embed_expanded, _ = self._expand_duration(
            duration_target, embeds_value, embeds_len, mels_len
        )

        mel_pred = self.mel_decoder(embed_expanded)
        mel_post = self.post_net(mel_pred)

        mel_select = (~mel_mask).unsqueeze(-1)
        mel_pred_flat = mel_pred.masked_select(mel_select)
        mel_post_flat = mel_post.masked_select(mel_select)
        mel_target_flat = mels.masked_select(mel_select).detach()

        embed_select = ~embeds_mask
        duration_target_flat = torch.log(
            duration_target.masked_select(embed_select) + 1.0
        )
        duration_pred_flat = torch.log(
            torch.clamp(duration_pred.masked_select(embed_select), min=0.0) + 1.0
        )

        mel_pred_loss = F.mse_loss(mel_pred_flat, mel_target_flat)
        mel_post_loss = F.mse_loss(mel_post_flat, mel_target_flat)
        duration_loss = F.l1_loss(duration_pred_flat, duration_target_flat)

        return {
            "mel_pred_loss": mel_pred_loss,
            "mel_post_loss": mel_post_loss,
            "duration_loss": duration_loss,
            "duration_pred": duration_pred,
            "mel_pred": mel_pred,
            "mel_post": mel_post,
        }

    @torch.inference_mode()
    def infer(
        self,
        embeds: torch.Tensor,
        embeds_len: torch.Tensor,
        vocab_embeds: torch.Tensor,
    ):
        embeds = self._prepare_embeds(embeds, vocab_embeds)
        embeds_key, embeds_value = self.embed_encoder(embeds, embeds_len)
        embeds_mask = get_mask_from_lengths(embeds_len, max_len=embeds.size(1))

        duration_pred = self.duration_predictor(embeds_value, is_train=False)
        duration_pred = duration_pred.masked_fill(embeds_mask, 0.0)

        duration_step = torch.round(duration_pred)
        duration_step = torch.clamp(duration_step, min=1.0)
        duration_step = duration_step.masked_fill(embeds_mask, 0.0)
        mel_lens = duration_step.sum(dim=-1).long()

        embed_expanded, mel_lens = self._expand_duration(
            duration_step,
            embeds_value,
            embeds_len,
            mel_lens,
        )
        mel_pred = self.mel_decoder(embed_expanded)
        mel_post = self.post_net(mel_pred)

        return {
            "duration_pred": duration_pred,
            "duration_step": duration_step,
            "mel_pred": mel_pred,
            "mel_post": mel_post,
            "mel_lens": mel_lens,
        }
