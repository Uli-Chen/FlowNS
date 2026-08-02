#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
../../.venv/bin/python train.py train \
  --model bpr \
  --architecture gmf \
  --embedding-dim 32 \
  --batch-size 1024 \
  --num-negatives 4 \
  --learning-rate 0.001 \
  --optimizer adam \
  --regularization 1e-5 \
  --epochs 250 \
  --eval-mode list \
  --list-length 160 \
  --device auto \
  --output runs/bpr
