from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import asdict
from dataclasses import field
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, FineGrainedFP8Config

from .components.wav_decoder import WavDecoder
from .components.wav_encoder import WavEncoder
from .talker import Talker, TalkerConfig
from .utils.tokenizer import (
    CHAR_VOCAB_SIZE,
    extract_last_assistant_char_aligned_embeds,
    tokenize_mask_assistant,
)


@dataclass
class LlmSpokenModelConfig:
    @dataclass
    class PeftConfig:
        enabled: bool = False
        task_type: str = "CAUSAL_LM"
        r: int = 8
        lora_alpha: int = 16
        lora_dropout: float = 0.0
        target_modules: list[str] | None = None
        bias: str = "none"

        @classmethod
        def from_dict(cls, config: dict):
            return cls(
                enabled=bool(config.get("enabled", False)),
                task_type=str(config.get("task_type", "CAUSAL_LM")),
                r=int(config.get("r", 8)),
                lora_alpha=int(config.get("lora_alpha", 16)),
                lora_dropout=float(config.get("lora_dropout", 0.0)),
                target_modules=config.get("target_modules"),
                bias=str(config.get("bias", "none")),
            )

    model: str
    template: str | None
    talker: TalkerConfig
    vocos_model_id: str = "charactr/vocos-mel-24khz"
    peft: PeftConfig = field(default_factory=PeftConfig)

    @classmethod
    def from_dict(cls, config: dict):
        return cls(
            model=config["model"],
            template=config.get("template"),
            talker=TalkerConfig.from_dict(config["talker"]),
            vocos_model_id=config.get("vocos_model_id", "charactr/vocos-mel-24khz"),
            peft=cls.PeftConfig.from_dict(config.get("peft", {})),
        )

    @classmethod
    def from_yaml(cls, path: str):
        import yaml

        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)


class LlmSpokenModel(nn.Module):
    def __init__(self, config: LlmSpokenModelConfig, apply_peft: bool = True):
        super().__init__()
        self.config = config

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            # quantization_config = FineGrainedFP8Config()
        )
        self.peft_enabled = config.peft.enabled

        if self.peft_enabled and apply_peft:
            from peft import LoraConfig, TaskType, get_peft_model

            try:
                task_type = getattr(TaskType, config.peft.task_type.upper())
            except AttributeError as exc:
                raise ValueError(
                    f"Invalid PEFT task_type: {config.peft.task_type}"
                ) from exc

            peft_config = LoraConfig(
                task_type=task_type,
                r=config.peft.r,
                lora_alpha=config.peft.lora_alpha,
                lora_dropout=config.peft.lora_dropout,
                target_modules=config.peft.target_modules,
                bias=config.peft.bias,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        if config.template:
            self.tokenizer.chat_template = open(config.template).read()
        self.config.talker.char_vocab_size = CHAR_VOCAB_SIZE
        self.talker = Talker(
            config.talker, device=self.model.device, dtype=self.model.dtype
        )
        self.wav_encoder = WavEncoder(
            mel_bins=config.talker.mel_bins, dtype=self.model.dtype
        )
        self.wav_decoder = WavDecoder(config.vocos_model_id)

        if not self.peft_enabled:
            # setting model not trainable
            for param in self.model.parameters():
                param.requires_grad = False

        # setting wav decoder not trainable
        for param in self.wav_decoder.parameters():
            param.requires_grad = False

    def setup_for_training(self) -> None:
        if self.peft_enabled:
            self.model.train()
        else:
            self.model.eval()
        self.talker.train()
        self.wav_encoder.eval()
        self.wav_decoder.eval()

    def optimizer_param_groups(self) -> list[dict[str, list[torch.nn.Parameter]]]:
        param_groups = [{"params": list(self.talker.parameters())}]
        if self.peft_enabled:
            peft_params = [p for p in self.model.parameters() if p.requires_grad]
            if peft_params:
                param_groups.append({"params": peft_params})
        return param_groups

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

        if self.peft_enabled:
            self.model.save_pretrained(save_path / "peft")

    @classmethod
    def from_pretrained(cls, path: str | Path):
        load_path = Path(path)
        with (load_path / "config.yaml").open("r", encoding="utf-8") as f:
            import yaml

            config_dict = yaml.safe_load(f)
        config = LlmSpokenModelConfig.from_dict(config_dict)
        peft_path = load_path / "peft"
        instant = cls(
            config=config,
            apply_peft=not (
                config.peft.enabled and peft_path.exists() and peft_path.is_dir()
            ),
        )

        talker_weights = load_file(str(load_path / "talker.safetensors"))
        instant.talker.load_state_dict(talker_weights)

        if peft_path.exists() and peft_path.is_dir():
            from peft import PeftModel

            instant.model = PeftModel.from_pretrained(
                instant.model,
                str(peft_path),
                is_trainable=instant.peft_enabled,
            )
        return instant

    def forward(
        self,
        messages_batch: list[list[dict]],
        audio_batch: list[np.ndarray],
        output_audio_list: bool = False,
    ):
        model_device = next(self.model.parameters()).device
        use_no_grad_llm = not (
            self.peft_enabled and self.training and self.model.training
        )
        llm_context = torch.no_grad() if use_no_grad_llm else nullcontext()

        with llm_context:
            input_ids, attention_mask, assistant_mask = tokenize_mask_assistant(
                self.tokenizer, messages_batch
            )
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            assistant_mask = assistant_mask.to(model_device)
            input_embeds = self.model.get_input_embeddings()(input_ids)
            hidden_outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = hidden_outputs.hidden_states[-1]
            (
                assistant_embeds,
                assistant_embeds_length,
                assistant_char_ids,
            ) = extract_last_assistant_char_aligned_embeds(
                self.tokenizer,
                input_ids,
                hidden_states,
                assistant_mask,
            )

        with torch.no_grad():
            audio_mels, audio_mel_lens = cast(
                tuple[torch.Tensor, torch.Tensor],
                self.wav_encoder.encode(audio_batch, model_device, return_audio=False),
            )

        outs = self.talker(
            assistant_embeds,
            assistant_embeds_length,
            assistant_char_ids,
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
