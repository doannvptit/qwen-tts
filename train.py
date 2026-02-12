import argparse
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
import wandb
import yaml
from datasets import Audio, concatenate_datasets, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from src.model import LlmSpokenModel, LlmSpokenModelConfig

SYSTEM_PROMPT = "say exactly provided sentence"
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
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    keep_last_n_checkpoints: int
    audio_dump_steps: int
    audio_dump_num_samples: int
    seed: int


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
        default="trainings/qwen3_4b.yaml",
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
        warmup_ratio=float(train_raw["warmup_ratio"]),
        logging_steps=int(train_raw["logging_steps"]),
        save_steps=int(train_raw["save_steps"]),
        keep_last_n_checkpoints=int(train_raw.get("keep_last_n_checkpoints", 0)),
        audio_dump_steps=int(train_raw.get("audio_dump_steps", 0)),
        audio_dump_num_samples=int(train_raw.get("audio_dump_num_samples", 2)),
        seed=int(train_raw["seed"]),
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
        text = str(sample[TEXT_COLUMN])
        messages_batch.append(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
                {"role": "assistant", "content": text},
            ]
        )
        audio_batch.append(_audio_to_numpy(sample[AUDIO_COLUMN]))

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
) -> None:
    ckpt_dir = output_dir / f"step_{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_dir / "model")

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        },
        ckpt_dir / "trainer_state.pt",
    )

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

    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg.training.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = LlmSpokenModelConfig.from_yaml(cfg.model_config)
    model = LlmSpokenModel(model_cfg)
    model.to(device)
    model.model.eval()
    model.talker.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params={trainable_params:,}")

    train_dataset = load_train_dataset(cfg.datasets, cfg.training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=False,
    )

    optimizer = AdamW(
        model.talker.parameters(),
        lr=cfg.training.learning_rate,
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

    wandb.init(
        project="llm-spoken",
        name=f"train-{config_path.stem}",
        config=raw_cfg,
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.training.num_train_epochs):
        progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.num_train_epochs}"
        )
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
                output_audio_list=should_dump_audio,
            )

            total_loss = (
                outputs["mel_pred_loss"]
                + outputs["mel_post_loss"]
                + outputs["duration_loss"]
            )
            (total_loss / cfg.training.gradient_accumulation_steps).backward()

            if step % cfg.training.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                log_payload = {
                    "loss/total": float(total_loss.item()),
                    "loss/mel_pred": float(outputs["mel_pred_loss"].item()),
                    "loss/mel_post": float(outputs["mel_post_loss"].item()),
                    "loss/duration": float(outputs["duration_loss"].item()),
                    "train/epoch": epoch + 1,
                    "train/step": global_step,
                    "train/lr": float(scheduler.get_last_lr()[0]),
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
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    save_checkpoint(
        output_dir=output_dir,
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=cfg.training.num_train_epochs,
        keep_last_n_checkpoints=cfg.training.keep_last_n_checkpoints,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
