#!/bin/bash
WANDB_MODE=online PYTHONUNBUFFERED=1 nohup $@ > training.log 2>&1 &