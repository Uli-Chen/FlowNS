#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
flow_run="${project_root}/experiments/flow_rns/p1_flow_finetune_lr5e5_seed1"

exec "${project_root}/../.venv/bin/python" "${project_root}/flow.py" rank-continuous \
  --flow-checkpoint "${flow_run}/best.pt" \
  --output "${project_root}/experiments/flow_rns/p2_uniform_trainable_seed1" \
  --train-item-embeddings \
  --batch-size 2048 \
  --learning-rate 0.001 \
  --discriminator-reg 0.00001 \
  --epochs 30 \
  --minimum-epochs 10 \
  --early-stopping 6 \
  --minimum-ndcg-improvement 0.00001 \
  --negative-source uniform \
  --selection-mode random \
  --recommendation-list-length 160 \
  --eval-batch-size 1024 \
  --diagnostics-examples 1024 \
  --seed 1 \
  --device cuda \
  --amp
