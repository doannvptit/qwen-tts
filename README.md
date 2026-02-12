# llm-spoken

`llm-spoken` is an experimental project to make an LLM speak directly from its hidden embeddings.

The core idea is simple: instead of generating text first and then sending text to a TTS model, we train a lightweight speech module (`Talker`) that maps assistant token embeddings from a frozen LLM into mel-spectrograms, then vocodes them into waveform audio.

This is one practical step toward a full-duplex speech model.

## What this project does

- Uses a frozen base LLM (`Qwen3-4B-Instruct` by default) to produce assistant hidden states.
- Trains only the speech head (`Talker`) to predict mel frames and durations.
- Uses `vocos` to decode predicted mels into waveform audio for qualitative checks.
- Supports training from one or multiple Hugging Face datasets.

## High-level pipeline

1. Build chat messages with a system prompt and target assistant text.
2. Run the frozen LLM and extract the last assistant embeddings.
3. Encode target waveform into mel-spectrogram (for supervision).
4. Train `Talker` to predict duration + mel from LLM embeddings.
5. Optionally decode predicted mel to `.wav` samples during training.

## Project structure

```text
llm-spoken/
├── train.py                    # Main training entrypoint
├── pyproject.toml              # Python project and dependencies
├── trainings/
│   └── qwen3_4b.yaml           # Training run config (datasets + optimizer + output)
├── configs/
│   ├── qwen3_4b.yaml           # Model config (base LLM + Talker + vocoder)
│   └── qwen3.jinja             # Chat template
├── src/
│   ├── model.py                # LlmSpokenModel wrapper (LLM + Talker + wav I/O)
│   ├── talker.py               # Trainable speech head
│   ├── components/             # Duration/mel encoder-decoder building blocks
│   ├── modules/                # Shared neural modules
│   └── utils/                  # Tokenization and utility helpers
├── checkpoints/                # Training outputs (ignored by git)
└── wandb/                      # Weights & Biases logs (ignored by git)
```

## Quick start

### 1) Install dependencies

This project requires Python `>=3.13`.

Using `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Configure training

- Edit dataset + training settings in `trainings/qwen3_4b.yaml`.
- Edit model + Talker settings in `configs/qwen3_4b.yaml`.

### 3) Run training

```bash
python train.py --config trainings/qwen3_4b.yaml
```

## Training outputs

- Checkpoints are saved to `training.output_dir` (default: `checkpoints/qwen3_4b`).
- Model snapshot includes:
  - `model/config.yaml`
  - `model/talker.safetensors`
  - `trainer_state.pt`
- If `audio_dump_steps > 0`, sample audio pairs are dumped at:
  - `checkpoints/.../step_x/audios/sample_i_pred.wav`
  - `checkpoints/.../step_x/audios/sample_i_target.wav`

## Current scope

This repository currently focuses on training the speech head from teacher-forced assistant text.

It is not yet a production real-time, streaming, fully full-duplex system, but it is designed as a clean foundation for that direction.
