from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .components.wav_decoder import WavDecoder
from .components.wav_encoder import WavEncoder
from .talker import Talker, TalkerConfig
from .utils.tokenizer import extract_last_assistant_embeds, tokenize_mask_assistant


@dataclass
class LlmSpokenModelConfig:
    model: str
    template: str | None
    talker: TalkerConfig
    vocos_model_id: str = "charactr/vocos-mel-24khz"

    @classmethod
    def from_yaml(cls, path: str):
        import yaml

        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(
            model=config["model"],
            template=config.get("template"),
            talker=TalkerConfig.from_dict(config["talker"]),
            vocos_model_id=config.get("vocos_model_id", "charactr/vocos-mel-24khz"),
        )


class LlmSpokenModel(nn.Module):
    def __init__(self, config: LlmSpokenModelConfig):
        super().__init__()
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(config.model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        if config.template:
            self.tokenizer.chat_template = open(config.template).read()
        self.talker = Talker(
            config.talker, device=self.model.device, dtype=self.model.dtype
        )
        self.wav_encoder = WavEncoder(
            mel_bins=config.talker.mel_bins, dtype=self.model.dtype
        )
        self.wav_decoder = WavDecoder(config.vocos_model_id)

        # setting model not trainable
        for param in self.model.parameters():
            param.requires_grad = False

        # setting wav decoder not trainable
        for param in self.wav_decoder.parameters():
            param.requires_grad = False

    def save_pretrained(self, path: str | Path):
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        with (save_path / "config.yaml").open("w", encoding="utf-8") as f:
            import yaml

            yaml.safe_dump(asdict(self.config), f)

        talker_state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.talker.state_dict().items()
        }
        save_file(talker_state, str(save_path / "talker.safetensors"))

    @classmethod
    def from_pretrained(cls, path: str | Path):
        load_path = Path(path)
        with (load_path / "config.yaml").open("r", encoding="utf-8") as f:
            import yaml

            config_dict = yaml.safe_load(f)
        config = LlmSpokenModelConfig(
            model=config_dict["model"],
            template=config_dict.get("template"),
            talker=TalkerConfig.from_dict(config_dict["talker"]),
            vocos_model_id=config_dict.get(
                "vocos_model_id", "charactr/vocos-mel-24khz"
            ),
        )
        instant = cls(config=config)

        talker_weights = load_file(str(load_path / "talker.safetensors"))
        instant.talker.load_state_dict(talker_weights)
        return instant

    def forward(
        self,
        messages_batch: list[list[dict]],
        audio_batch: list[np.ndarray],
        output_audio_list: bool = False,
    ):
        model_device = next(self.model.parameters()).device
        with torch.no_grad():
            input_ids, attention_mask, assistant_mask = tokenize_mask_assistant(
                self.tokenizer, messages_batch
            )
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            assistant_mask = assistant_mask.to(model_device)
            input_embeds = self.model.get_input_embeddings()(input_ids)
            hidden_outputs = self.model.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
            )
            hidden_states = hidden_outputs.last_hidden_state
            assistant_embeds, assistant_embeds_length = extract_last_assistant_embeds(
                hidden_states, assistant_mask
            )

        with torch.no_grad():
            audio_mels, audio_mel_lens = cast(
                tuple[torch.Tensor, torch.Tensor],
                self.wav_encoder.encode(audio_batch, model_device, return_audio=False),
            )

        outs = self.talker(
            assistant_embeds,
            assistant_embeds_length,
            audio_mels,
            audio_mel_lens,
        )

        if output_audio_list:
            mel_post = outs["mel_post"].to(dtype=torch.float32)
            self.wav_decoder = self.wav_decoder.to(mel_post.device)
            audio_preds, audio_lens = self.wav_decoder.decode(mel_post, audio_mel_lens)
            outs["audio_list"] = [
                audio_preds[i, : int(audio_lens[i].item())].detach().cpu().numpy()
                for i in range(audio_preds.size(0))
            ]
        return outs
