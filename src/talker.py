from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .components.duration_aligner import DurationAligner
from .components.duration_predictor import DurationPredictor
from .components.mel_decoder import MelDecoder
from .components.mel_encoder import MelEncoder
from .components.post_net import PostNet
from .utils.tools import get_mask_from_lengths


@dataclass
class TalkerConfig:
    input_dim: int
    hidden_dim: int
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

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class Talker(nn.Module):
    def __init__(self, cfg: TalkerConfig, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.projection = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.duration_predictor = DurationPredictor(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            k_size=cfg.duration_predictor_kernel_size,
            layers=cfg.duration_predictor_num_layers,
            dropout_rate=cfg.dropout_rate,
        )
        self.duration_aligner = DurationAligner(
            input_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim
        )
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
        self.to(device=device, dtype=dtype)

    def _expand_duration(
        self,
        duration_target: torch.Tensor,
        embeds: torch.Tensor,
        embeds_len: torch.Tensor,
        mel_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, hidden_dim = embeds.shape
        max_mel_len = int(mel_lens.max().item())
        expanded = embeds.new_zeros((batch_size, max_mel_len, hidden_dim))

        duration_steps = torch.clamp(torch.round(duration_target), min=0).to(
            dtype=torch.long
        )

        for i in range(batch_size):
            valid_tokens = int(embeds_len[i].item())
            target_mel_len = int(mel_lens[i].item())
            token_embeds = embeds[i, :valid_tokens]
            token_durations = duration_steps[i, :valid_tokens]

            repeated = torch.repeat_interleave(token_embeds, token_durations, dim=0)
            if repeated.size(0) == 0:
                repeated = embeds.new_zeros((target_mel_len, hidden_dim))

            if repeated.size(0) < target_mel_len:
                pad = embeds.new_zeros((target_mel_len - repeated.size(0), hidden_dim))
                repeated = torch.cat([repeated, pad], dim=0)
            else:
                repeated = repeated[:target_mel_len]

            expanded[i, :target_mel_len] = repeated

        return expanded, mel_lens

    def forward(
        self,
        embeds: torch.Tensor,
        embeds_len: torch.Tensor,
        mels: torch.Tensor,
        mels_len: torch.Tensor,
    ):
        mel_mask = get_mask_from_lengths(mels_len, max_len=mels.size(1))
        embeds = self.projection(embeds)
        embeds_mask = get_mask_from_lengths(embeds_len, max_len=embeds.size(1))

        mel_h = self.mel_encoder(mels)
        duration_target = self.duration_aligner(embeds, embeds_len, mel_h, mels_len)
        duration_target = duration_target.masked_fill(embeds_mask, 0.0)
        duration_pred = self.duration_predictor(embeds)

        embed_expanded, _ = self._expand_duration(
            duration_target, embeds, embeds_len, mels_len
        )

        mel_pred = self.mel_decoder(embed_expanded)
        mel_post = self.post_net(mel_pred)

        mel_select = (~mel_mask).unsqueeze(-1)
        mel_pred_flat = mel_pred.masked_select(mel_select)
        mel_post_flat = mel_post.masked_select(mel_select)
        mel_target_flat = mels.masked_select(mel_select).detach()

        text_select = ~embeds_mask
        duration_target_flat = torch.log(
            duration_target.masked_select(text_select) + 1.0
        )
        duration_pred_flat = torch.log(
            torch.clamp(duration_pred.masked_select(text_select), min=0.0) + 1.0
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
