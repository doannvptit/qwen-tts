import argparse
import math
import os
os.environ["HF_DATASETS_CACHE"] = "/data"
import random
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
import wandb
import yaml
from datasets import Audio, concatenate_datasets, load_dataset
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import get_scheduler

from src.model import LlmSpokenModel, LlmSpokenModelConfig

SYSTEM_PROMPTS = [
    "say exactly provided sentence",
    "repeat the following sentence",
    "please say the following sentence",
    "say the sentence exactly as given",
    "repeat exactly what is written below",
    "please repeat the sentence verbatim",
    "say the following text exactly",
    "repeat the provided sentence exactly",
    "speak the sentence exactly as written",
    "read and repeat the following sentence",
    "please repeat exactly the given sentence",
    "say precisely the provided sentence",
    "repeat word for word the following sentence",
    "please say exactly what is written",
    "read the sentence and repeat it exactly",
    "repeat the sentence without any changes",
    "say the exact sentence provided",
    "please repeat the exact sentence below",
    "repeat exactly the sentence given",
    "speak exactly the following sentence",
]
TEXT_COLUMN = "transcription"
AUDIO_COLUMN = "audio"
DEFAULT_SPLIT = "train"


@dataclass
class DatasetSpec:
    name: str
    split: str
    subset: str | None
    text_column: str
    audio_column: str


@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    llm_loss_weight: float
    tts_loss_weight: float
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    keep_last_n_checkpoints: int
    audio_dump_steps: int
    audio_dump_num_samples: int
    seed: int
    discriminator_learning_rate: float
    discriminator_warmup_steps: int
    discriminator_loss_weight: float
    feature_matching_loss_weight: float


@dataclass
class RunConfig:
    datasets: list[DatasetSpec]
    model_config: str
    training: TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="trainings/Qwen3-0.6B-Instruct-freeze.yaml",
        help="Path to training yaml config.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_run_config(config_path: Path) -> tuple[RunConfig, dict]:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    datasets_cfg = []
    for dataset_raw in raw["datasets"]:
        datasets_cfg.append(
            DatasetSpec(
                name=dataset_raw["name"],
                split=str(dataset_raw.get("split", DEFAULT_SPLIT)),
                subset=dataset_raw.get("subset"),
                text_column=str(dataset_raw.get("text_column", TEXT_COLUMN)),
                audio_column=str(dataset_raw.get("audio_column", AUDIO_COLUMN)),
            )
        )
    train_raw = raw["training"]
    training_cfg = TrainingConfig(
        output_dir=str(train_raw["output_dir"]),
        num_train_epochs=int(train_raw["num_train_epochs"]),
        per_device_train_batch_size=int(train_raw["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_raw["gradient_accumulation_steps"]),
        learning_rate=float(train_raw["learning_rate"]),
        weight_decay=float(train_raw["weight_decay"]),
        llm_loss_weight=float(train_raw.get("llm_loss_weight", 1.0)),
        tts_loss_weight=float(train_raw.get("tts_loss_weight", 1.0)),
        warmup_ratio=float(train_raw["warmup_ratio"]),
        logging_steps=int(train_raw["logging_steps"]),
        save_steps=int(train_raw["save_steps"]),
        keep_last_n_checkpoints=int(train_raw.get("keep_last_n_checkpoints", 0)),
        audio_dump_steps=int(train_raw.get("audio_dump_steps", 0)),
        audio_dump_num_samples=int(train_raw.get("audio_dump_num_samples", 2)),
        seed=int(train_raw["seed"]),
        discriminator_learning_rate=float(
            train_raw.get("discriminator_learning_rate", 2e-4)
        ),
        discriminator_warmup_steps=int(
            train_raw.get("discriminator_warmup_steps", 10000)
        ),
        discriminator_loss_weight=float(
            train_raw.get("discriminator_loss_weight", 1.0)
        ),
        feature_matching_loss_weight=float(
            train_raw.get("feature_matching_loss_weight", 1.0)
        ),
    )
    if training_cfg.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if training_cfg.logging_steps < 1:
        raise ValueError("logging_steps must be >= 1")
    if training_cfg.save_steps < 1:
        raise ValueError("save_steps must be >= 1")
    if training_cfg.keep_last_n_checkpoints < 0:
        raise ValueError("keep_last_n_checkpoints must be >= 0")
    if training_cfg.audio_dump_steps < 0:
        raise ValueError("audio_dump_steps must be >= 0")
    if training_cfg.audio_dump_num_samples < 1:
        raise ValueError("audio_dump_num_samples must be >= 1")
    if training_cfg.llm_loss_weight < 0:
        raise ValueError("llm_loss_weight must be >= 0")
    if training_cfg.tts_loss_weight < 0:
        raise ValueError("tts_loss_weight must be >= 0")
    if training_cfg.llm_loss_weight == 0 and training_cfg.tts_loss_weight == 0:
        raise ValueError(
            "At least one of llm_loss_weight or tts_loss_weight must be > 0"
        )
    if training_cfg.discriminator_warmup_steps < 0:
        raise ValueError("discriminator_warmup_steps must be >= 0")
    if training_cfg.discriminator_loss_weight < 0:
        raise ValueError("discriminator_loss_weight must be >= 0")
    if training_cfg.feature_matching_loss_weight < 0:
        raise ValueError("feature_matching_loss_weight must be >= 0")
    if training_cfg.discriminator_learning_rate < 0:
        raise ValueError("discriminator_learning_rate must be >= 0")

    raw_model_config_path = Path(raw["model_config"])
    if raw_model_config_path.is_absolute():
        model_config_path = raw_model_config_path
    else:
        candidate_from_config = (config_path.parent / raw_model_config_path).resolve()
        if candidate_from_config.exists():
            model_config_path = candidate_from_config
        else:
            model_config_path = raw_model_config_path.resolve()
    cfg = RunConfig(
        datasets=datasets_cfg,
        model_config=str(model_config_path),
        training=training_cfg,
    )
    return cfg, raw


def load_train_dataset(dataset_specs: list[DatasetSpec], seed: int):
    datasets = []
    for spec in dataset_specs:
        ds = load_dataset(spec.name, name=spec.subset, split=spec.split).cast_column(
            spec.audio_column, Audio(sampling_rate=24_000)
        )
        if spec.text_column != TEXT_COLUMN:
            ds = ds.rename_column(spec.text_column, TEXT_COLUMN)
        if spec.audio_column != AUDIO_COLUMN:
            ds = ds.rename_column(spec.audio_column, AUDIO_COLUMN)
        datasets.append(ds)
    merged = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    return merged.shuffle(seed=seed)


def _audio_to_numpy(audio) -> np.ndarray:
    return audio["array"]


def collate_batch(samples: list[dict]) -> dict:
    messages_batch = []
    audio_batch = []

    for sample in samples:
        audio = _audio_to_numpy(sample[AUDIO_COLUMN])
        text = str(sample[TEXT_COLUMN])

        if len(text) < 10 or len(audio) < 24000:
            print(f"Skipping sample with text length {len(text)} and audio length {len(audio)}")
            continue
        messages_batch.append(
            [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                {"role": "user", "content": text},
                {"role": "assistant", "content": text.lower().strip()}, # TODO: is this necessary to lowercase?
            ]
        )
        audio_batch.append(audio)

    return {
        "messages_batch": messages_batch,
        "audio_batch": audio_batch,
    }


def save_checkpoint(
    output_dir: Path,
    global_step: int,
    model: LlmSpokenModel,
    optimizer: AdamW,
    scheduler,
    epoch: int,
    keep_last_n_checkpoints: int,
    disc_optimizer: AdamW | None = None,
    disc_scheduler=None,
) -> None:
    ckpt_dir = output_dir / f"step_{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.save_pretrained(ckpt_dir / "model")

    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
    }
    if disc_optimizer is not None:
        state["disc_optimizer"] = disc_optimizer.state_dict()
    if disc_scheduler is not None:
        state["disc_scheduler"] = disc_scheduler.state_dict()
    torch.save(state, ckpt_dir / "trainer_state.pt")

    if keep_last_n_checkpoints > 0:
        checkpoint_dirs: list[tuple[int, Path]] = []
        for path in output_dir.glob("step_*"):
            if not path.is_dir():
                continue
            if not path.name.startswith("step_"):
                continue
            step_str = path.name[len("step_") :]
            if not step_str.isdigit():
                continue
            checkpoint_dirs.append((int(step_str), path))

        checkpoint_dirs.sort(key=lambda item: item[0])
        stale_dirs = checkpoint_dirs[:-keep_last_n_checkpoints]
        for _, stale_dir in stale_dirs:
            shutil.rmtree(stale_dir, ignore_errors=False)


def save_debug_audios(
    output_dir: Path,
    global_step: int,
    predicted_audio_list: list[np.ndarray],
    target_audio_list: list[np.ndarray],
    max_samples: int,
    sample_rate: int = 24_000,
) -> None:
    dump_dir = output_dir / f"step_{global_step}" / "audios"
    dump_dir.mkdir(parents=True, exist_ok=True)

    sample_count = min(max_samples, len(predicted_audio_list), len(target_audio_list))
    for i in range(sample_count):
        pred_audio = torch.as_tensor(
            predicted_audio_list[i], dtype=torch.float32
        ).unsqueeze(0)
        target_audio = torch.as_tensor(
            target_audio_list[i], dtype=torch.float32
        ).unsqueeze(0)
        torchaudio.save(
            str(dump_dir / f"sample_{i}_pred.wav"), pred_audio.cpu(), sample_rate
        )
        torchaudio.save(
            str(dump_dir / f"sample_{i}_target.wav"),
            target_audio.cpu(),
            sample_rate,
        )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg, raw_cfg = load_run_config(config_path)

    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = rank == 0
    set_seed(cfg.training.seed + rank)

    if is_main:
        output_dir = Path(cfg.training.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed:
        dist.barrier()

    output_dir = Path(cfg.training.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = LlmSpokenModelConfig.from_yaml(cfg.model_config)
    model = LlmSpokenModel(model_cfg)
    model.to(device)
    model.setup_for_training()

    use_llm_loss = model.peft_enabled
    effective_llm_loss_weight = cfg.training.llm_loss_weight if use_llm_loss else 0.0
    if is_main and cfg.training.llm_loss_weight > 0 and not use_llm_loss:
        print("LLM is frozen (PEFT disabled): llm_loss_weight is ignored.")
    if effective_llm_loss_weight == 0 and cfg.training.tts_loss_weight == 0:
        raise ValueError(
            "No trainable objective: tts_loss_weight and effective llm_loss_weight are both 0"
        )

    if is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"trainable_params={trainable_params:,}")
        print(
            f"distributed: world_size={world_size} rank={rank} local_rank={local_rank}"
        )

    train_dataset = load_train_dataset(cfg.datasets, cfg.training.seed + rank)
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.training.seed,
            drop_last=False,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.per_device_train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_batch,
        drop_last=False,
    )

    optimizer = AdamW(
        raw_model.optimizer_param_groups(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    disc_optimizer = AdamW(
        raw_model.discriminator_param_groups(),
        lr=cfg.training.discriminator_learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / max(cfg.training.gradient_accumulation_steps, 1)
    )
    total_training_steps = cfg.training.num_train_epochs * updates_per_epoch
    warmup_steps = int(total_training_steps * cfg.training.warmup_ratio)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    disc_warmup_steps = max(
        cfg.training.discriminator_warmup_steps - warmup_steps, 0
    )
    disc_scheduler = get_scheduler(
        name="linear",
        optimizer=disc_optimizer,
        num_warmup_steps=disc_warmup_steps,
        num_training_steps=total_training_steps,
    )

    if is_main:
        wandb.init(
            project="llm-spoken",
            name=f"train-{config_path.stem}",
            config=raw_cfg,
        )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    disc_optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.training.num_train_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        if is_main:
            progress = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.num_train_epochs}"
            )
        else:
            progress = train_loader

        for step, batch in enumerate(progress, start=1):
            will_update = step % cfg.training.gradient_accumulation_steps == 0
            next_global_step = global_step + 1 if will_update else global_step
            should_dump_audio = (
                will_update
                and cfg.training.audio_dump_steps > 0
                and next_global_step % cfg.training.audio_dump_steps == 0
            )

            outputs = model(
                messages_batch=batch["messages_batch"],
                audio_batch=batch["audio_batch"],
                output_audio_list=should_dump_audio and is_main,
            )

            tts_loss = (
                outputs["mel_pred_loss"]
                + outputs["mel_post_loss"]
                + outputs["duration_loss"]
            )
            llm_loss = outputs["llm_loss"]
            adv_loss = outputs["adv_loss"]
            feat_match_loss = outputs["feat_match_loss"]
            weighted_tts_loss = cfg.training.tts_loss_weight * tts_loss
            weighted_llm_loss = effective_llm_loss_weight * llm_loss
            weighted_adv_loss = cfg.training.discriminator_loss_weight * adv_loss
            weighted_fm_loss = (
                cfg.training.feature_matching_loss_weight * feat_match_loss
            )
            total_loss = (
                weighted_tts_loss
                + weighted_llm_loss
                + weighted_adv_loss
                + weighted_fm_loss
            )
            if is_main:
                print(
                    f"Llm Loss: {llm_loss.item()}, Tts Loss: {tts_loss.item()}, "
                    f"Adv Loss: {adv_loss.item()}, FM Loss: {feat_match_loss.item()}, "
                    f"Total Loss: {total_loss.item()}"
                )

            d_loss = raw_model.discriminator_step(
                outputs["mel_post"], outputs["audio_mels"]
            )

            sync_context = (
                model.no_sync() if (is_distributed and not will_update) else nullcontext()
            )
            with sync_context:
                if d_loss is not None:
                    (d_loss / cfg.training.gradient_accumulation_steps).backward()
                (total_loss / cfg.training.gradient_accumulation_steps).backward()

            if will_update:
                torch.nn.utils.clip_grad_norm_(
                    raw_model.generator_parameters(), 1.0
                )
                torch.nn.utils.clip_grad_norm_(
                    raw_model.discriminator.parameters(), 1.0
                )
                optimizer.step()
                disc_optimizer.step()
                scheduler.step()
                disc_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                disc_optimizer.zero_grad(set_to_none=True)
                global_step += 1
                raw_model.disc_step.fill_(global_step)

                if is_main:
                    log_payload = {
                        "loss/total": float(total_loss.item()),
                        "loss/tts": float(tts_loss.item()),
                        "loss/llm": float(llm_loss.item()),
                        "loss/weighted_tts": float(weighted_tts_loss.item()),
                        "loss/weighted_llm": float(weighted_llm_loss.item()),
                        "loss/adv": float(adv_loss.item()),
                        "loss/feat_match": float(feat_match_loss.item()),
                        "loss/weighted_adv": float(weighted_adv_loss.item()),
                        "loss/weighted_fm": float(weighted_fm_loss.item()),
                        "loss/d": float(d_loss.item()) if d_loss is not None else 0.0,
                        "train/use_llm_loss": float(use_llm_loss),
                        "train/disc_active": float(outputs["disc_active"]),
                        "loss/mel_pred": float(outputs["mel_pred_loss"].item()),
                        "loss/mel_post": float(outputs["mel_post_loss"].item()),
                        "loss/duration": float(outputs["duration_loss"].item()),
                        "train/epoch": epoch + 1,
                        "train/step": global_step,
                        "train/lr": float(scheduler.get_last_lr()[0]),
                        "train/disc_lr": float(disc_scheduler.get_last_lr()[0]),
                    }
                    wandb.log(log_payload, step=global_step)

                    progress.set_postfix(
                        loss=f"{log_payload['loss/total']:.4f}",
                        step=global_step,
                    )

                    if global_step % cfg.training.logging_steps == 0:
                        print(
                            f"step={global_step} "
                            f"loss={log_payload['loss/total']:.4f} "
                            f"tts={log_payload['loss/tts']:.4f} "
                            f"llm={log_payload['loss/llm']:.4f} "
                            f"adv={log_payload['loss/adv']:.4f} "
                            f"fm={log_payload['loss/feat_match']:.4f} "
                            f"d={log_payload['loss/d']:.4f} "
                            f"mel_pred={log_payload['loss/mel_pred']:.4f} "
                            f"mel_post={log_payload['loss/mel_post']:.4f} "
                            f"duration={log_payload['loss/duration']:.4f} "
                            f"lr={log_payload['train/lr']:.6e}"
                        )

                    if global_step % cfg.training.save_steps == 0:
                        save_checkpoint(
                            output_dir,
                            global_step,
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            cfg.training.keep_last_n_checkpoints,
                            disc_optimizer=disc_optimizer,
                            disc_scheduler=disc_scheduler,
                        )

                    if should_dump_audio:
                        print("Dumping audio...")
                        save_debug_audios(
                            output_dir=output_dir,
                            global_step=global_step,
                            predicted_audio_list=outputs["audio_list"],
                            target_audio_list=batch["audio_batch"],
                            max_samples=cfg.training.audio_dump_num_samples,
                        )

        if len(train_loader) % cfg.training.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                raw_model.generator_parameters(), 1.0
            )
            torch.nn.utils.clip_grad_norm_(
                raw_model.discriminator.parameters(), 1.0
            )
            optimizer.step()
            disc_optimizer.step()
            scheduler.step()
            disc_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            disc_optimizer.zero_grad(set_to_none=True)
            global_step += 1
            raw_model.disc_step.fill_(global_step)

    if is_main:
        save_checkpoint(
            output_dir=output_dir,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=cfg.training.num_train_epochs,
            keep_last_n_checkpoints=cfg.training.keep_last_n_checkpoints,
            disc_optimizer=disc_optimizer,
            disc_scheduler=disc_scheduler,
        )
        wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
