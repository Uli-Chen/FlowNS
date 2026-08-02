#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
../../.venv/bin/python train.py train \
  --model kbgan \
  --architecture gmf \
  --pretrained-checkpoint checkpoints/pretrain_model_dis_zhihu.pkl \
  --embedding-dim 32 \
  --batch-size 1024 \
  --candidates 30 \
  --alpha 0 \
  --learning-rate 0.001 \
  --optimizer adam \
  --regularization 1e-5 \
  --generator-regularization 1e-5 \
  --epochs 400 \
  --eval-mode list \
  --list-length 160 \
  --device auto \
  --output runs/kbgan
