#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${project_root}/../.venv/bin/python" "${project_root}/flow.py" pretrain \
  --output "${project_root}/experiments/flow_rns/p1_flow_pretrain_w512_d8_seed1" \
  --hidden-dim 512 \
  --depth 8 \
  --user-dim 128 \
  --time-dim 128 \
  --sampling-steps 16 \
  --reconstruction-weight 0.25 \
  --batch-size 2048 \
  --learning-rate 0.0002 \
  --weight-decay 0.00001 \
  --epochs 50 \
  --minimum-epochs 10 \
  --early-stopping 5 \
  --minimum-relative-improvement 0.005 \
  --validation-fraction 0.02 \
  --validation-examples 131072 \
  --diagnostics-examples 4096 \
  --diagnostics-projections 64 \
  --gradient-clip 1.0 \
  --seed 1 \
  --device cuda \
  --amp
