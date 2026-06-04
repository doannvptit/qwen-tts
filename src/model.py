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

from .components.mel_discriminator import (
    MultiScaleMelDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)
from .components.wav_decoder import WavDecoder
from .components.wav_encoder import WavEncoder
from .talker import Talker, TalkerConfig
from .utils.tokenizer import (
    CHAR_VOCAB_SIZE,
    build_assistant_labels,
    expand_token_embeds_to_chars,
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
    discriminator_warmup_steps: int = 10000
    discriminator_loss_weight: float = 1.0
    feature_matching_loss_weight: float = 1.0

    @classmethod
    def from_dict(cls, config: dict):
        return cls(
            model=config["model"],
            template=config.get("template"),
            talker=TalkerConfig.from_dict(config["talker"]),
            vocos_model_id=config.get("vocos_model_id", "charactr/vocos-mel-24khz"),
            peft=cls.PeftConfig.from_dict(config.get("peft", {})),
            discriminator_warmup_steps=int(
                config.get("discriminator_warmup_steps", 10000)
            ),
            discriminator_loss_weight=float(
                config.get("discriminator_loss_weight", 1.0)
            ),
            feature_matching_loss_weight=float(
                config.get("feature_matching_loss_weight", 1.0)
            ),
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
        self.discriminator = MultiScaleMelDiscriminator(n_mels=config.talker.mel_bins)
        self.discriminator.to(device=self.model.device, dtype=torch.float32)

        self.disc_warmup_steps = int(config.discriminator_warmup_steps)
        self.disc_loss_weight = float(config.discriminator_loss_weight)
        self.feat_match_loss_weight = float(config.feature_matching_loss_weight)
        self.register_buffer("disc_step", torch.tensor(0, dtype=torch.long))

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
        self.discriminator.train()

    def optimizer_param_groups(self) -> list[dict[str, list[torch.nn.Parameter]]]:
        param_groups = [{"params": list(self.talker.parameters())}]
        if self.peft_enabled:
            peft_params = [p for p in self.model.parameters() if p.requires_grad]
            if peft_params:
                param_groups.append({"params": peft_params})
        return param_groups

    def discriminator_param_groups(self) -> list[dict[str, list[torch.nn.Parameter]]]:
        return [{"params": list(self.discriminator.parameters())}]

    def generator_parameters(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = list(self.talker.parameters())
        if self.peft_enabled:
            params.extend(p for p in self.model.parameters() if p.requires_grad)
        return params

    def is_discriminator_active(self) -> bool:
        return int(self.disc_step.item()) >= self.disc_warmup_steps

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

        discriminator_state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.discriminator.state_dict().items()
        }
        save_file(discriminator_state, str(save_path / "discriminator.safetensors"))

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

        discriminator_path = load_path / "discriminator.safetensors"
        if discriminator_path.exists():
            discriminator_weights = load_file(str(discriminator_path))
            instant.discriminator.load_state_dict(discriminator_weights)

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
        max_position_offset: int = 4096,
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
            labels = build_assistant_labels(input_ids, assistant_mask)
            input_embeds = self.model.get_input_embeddings()(input_ids)

            position_offset = int(
                torch.randint(0, max_position_offset + 1, (1,)).item()
            )
            position_ids = torch.arange(
                position_offset,
                position_offset + input_ids.size(1),
                device=model_device,
                dtype=torch.long,
            ).unsqueeze(0)

            hidden_outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )
            llm_loss = hidden_outputs.loss
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
            print(
                f"Assistant embeds shape: {assistant_embeds.shape}, assistant_embeds_length: {assistant_embeds_length}, assistant_char_ids shape: {assistant_char_ids.shape}"
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
        outs["llm_loss"] = llm_loss
        outs["audio_mels"] = audio_mels

        adv_loss = torch.zeros((), device=model_device, dtype=torch.float32)
        feat_match_loss = torch.zeros((), device=model_device, dtype=torch.float32)
        if self.training and self.is_discriminator_active():
            mel_post_f = outs["mel_post"].to(dtype=torch.float32)
            audio_mels_f = audio_mels.to(dtype=torch.float32)
            disc_fake_out, disc_fake_fmap = self.discriminator(mel_post_f)
            with torch.no_grad():
                disc_real_out, disc_real_fmap = self.discriminator(audio_mels_f)
            adv_loss = generator_adversarial_loss(disc_fake_out).to(
                device=model_device, dtype=torch.float32
            )
            feat_match_loss = feature_matching_loss(
                disc_real_fmap, disc_fake_fmap
            ).to(device=model_device, dtype=torch.float32)
        outs["adv_loss"] = adv_loss
        outs["feat_match_loss"] = feat_match_loss
        outs["disc_active"] = self.is_discriminator_active()

        if output_audio_list:
            mel_post = outs["mel_post"].to(dtype=torch.float32)
            self.wav_decoder = self.wav_decoder.to(mel_post.device)
            audio_preds, audio_lens = self.wav_decoder.decode(mel_post, audio_mel_lens)
            outs["audio_list"] = [
                audio_preds[i, : int(audio_lens[i].item())].detach().cpu().numpy()
                for i in range(audio_preds.size(0))
            ]
        return outs

    def discriminator_step(
        self,
        mel_post: torch.Tensor,
        audio_mels: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.is_discriminator_active():
            return None
        mel_post = mel_post.detach().to(dtype=torch.float32)
        audio_mels = audio_mels.detach().to(dtype=torch.float32)
        disc_real_out, _ = self.discriminator(audio_mels)
        disc_fake_out, _ = self.discriminator(mel_post)
        loss = discriminator_loss(disc_real_out, disc_fake_out)
        return loss

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        scaled_logits = logits / max(temperature, 1e-5)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            sorted_logits = sorted_logits.masked_fill(sorted_mask, -float("inf"))
            probs = torch.softmax(sorted_logits, dim=-1)
            sampled_sorted = torch.multinomial(probs, num_samples=1)
            return sorted_indices.gather(-1, sampled_sorted)

        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.inference_mode()
    def stream_generate_assistant(
        self,
        messages: list[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        sample_rate: int = 24_000,
        max_position_offset: int = 4096,
    ):
        model_device = next(self.model.parameters()).device
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(model_device)
        attention_mask = model_inputs["attention_mask"].to(model_device)

        position_offset = int(
            torch.randint(0, max_position_offset + 1, (1,)).item()
        )
        position_ids = torch.arange(
            position_offset,
            position_offset + input_ids.size(1),
            device=model_device,
            dtype=torch.long,
        ).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = self._sample_next_token(
            outputs.logits[:, -1, :], temperature, top_p
        )

        eos_token_id = self.tokenizer.eos_token_id
        generated_token_ids: list[int] = []
        generated_token_embeds: list[torch.Tensor] = []
        decoded_text = ""
        next_position = position_offset + input_ids.size(1)

        while (
            len(generated_token_ids) < max_new_tokens
            and next_token.item() != eos_token_id
        ):
            token_id = int(next_token.item())
            generated_token_ids.append(token_id)

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype, device=model_device),
                ],
                dim=1,
            )
            position_ids = torch.tensor(
                [[next_position]], dtype=torch.long, device=model_device
            )
            outputs = self.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            generated_token_embeds.append(outputs.hidden_states[-1][0, -1])
            next_position += 1

            delta = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if delta:
                decoded_text += delta
                yield {
                    "event": "text",
                    "text": decoded_text,
                }

            next_token = self._sample_next_token(
                outputs.logits[:, -1, :],
                temperature,
                top_p,
            )

        final_text = decoded_text.strip()
        if not final_text:
            final_text = "I could not generate a response. Please try again."

        if not generated_token_embeds:
            yield {
                "event": "final",
                "text": final_text,
                "audio": None,
                "sample_rate": sample_rate,
            }
            return

        token_embed_tensor = torch.stack(generated_token_embeds, dim=0)
        expanded_embeds, expanded_char_ids = expand_token_embeds_to_chars(
            self.tokenizer,
            generated_token_ids,
            token_embed_tensor,
        )
        assistant_embeds = expanded_embeds.unsqueeze(0)
        assistant_char_ids = expanded_char_ids.unsqueeze(0)
        assistant_embeds_len = torch.tensor(
            [expanded_embeds.size(0)],
            dtype=torch.long,
            device=expanded_embeds.device,
        )

        talker_outs = self.talker.infer(
            assistant_embeds,
            assistant_embeds_len,
            assistant_char_ids,
        )
        mel_post = talker_outs["mel_post"].to(dtype=torch.float32)
        mel_lens = talker_outs["mel_lens"]

        self.wav_decoder = self.wav_decoder.to(mel_post.device)
        audio_preds, audio_lens = self.wav_decoder.decode(mel_post, mel_lens)
        audio = audio_preds[0, : int(audio_lens[0].item())].detach().cpu().numpy()

        yield {
            "event": "final",
            "text": final_text,
            "audio": audio,
            "sample_rate": sample_rate,
        }
