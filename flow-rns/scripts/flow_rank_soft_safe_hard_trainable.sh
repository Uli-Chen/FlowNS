#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
flow_run="${project_root}/experiments/flow_rns/p1_flow_finetune_lr5e5_seed1"

exec "${project_root}/../.venv/bin/python" "${project_root}/flow.py" rank-continuous \
  --flow-checkpoint "${flow_run}/best.pt" \
  --output "${project_root}/experiments/flow_rns/p2_soft_safe_hard_trainable_seed1" \
  --train-item-embeddings \
  --batch-size 2048 \
  --learning-rate 0.001 \
  --discriminator-reg 0.00001 \
  --epochs 30 \
  --minimum-epochs 10 \
  --early-stopping 6 \
  --minimum-ndcg-improvement 0.00001 \
  --candidates-per-user 16 \
  --pool-batch-size 2048 \
  --pool-refresh-epochs 5 \
  --flow-steps 16 \
  --flow-solver heun \
  --negative-source soft-flow \
  --selection-mode safe-hard \
  --hardness-beta 2.0 \
  --minimum-ess-ratio 0.5 \
  --safety-temperature 1.0 \
  --transport-neighbors 8 \
  --transport-minimum-ess-ratio 0.5 \
  --transport-max-inverse-temperature 20.0 \
  --recommendation-list-length 160 \
  --eval-batch-size 1024 \
  --diagnostics-examples 1024 \
  --seed 1 \
  --device cuda \
  --amp
