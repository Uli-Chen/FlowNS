#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
../../.venv/bin/python train.py train \
  --model rns \
  --architecture gmf \
  --pretrained-checkpoint checkpoints/pretrain_model_dis_zhihu.pkl \
  --embedding-dim 32 \
  --batch-size 1024 \
  --learning-rate 0.001 \
  --optimizer adam \
  --regularization 1e-5 \
  --generator-regularization 1e-5 \
  --epochs 400 \
  --candidates 30 \
  --exposure-candidates 1 \
  --alpha 2.5 \
  --beta 0.75 \
  --sigma-range 19,20,21,22,23 \
  --eval-mode list \
  --list-length 160 \
  --early-stopping 10 \
  --device auto \
  --output runs/rns
