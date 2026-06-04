#!/bin/bash
# Launch multi-GPU training via torchrun.
# Usage:
#   ./run_train_ddp.sh trainings/Qwen3-0.6B-Instruct.yaml
#   NPROC=2 ./run_train_ddp.sh trainings/Qwen3-0.6B-Instruct.yaml
set -euo pipefail

CONFIG="${1:-trainings/Qwen3-4B-Instruct-freeze.yaml}"
NPROC="${NPROC:-2}"

export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

torchrun --standalone --nproc_per_node="${NPROC}" train.py --config "${CONFIG}"
