#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="${project_root}/experiments/flow_rns/p1_flow_finetune_lr5e5_seed1"

exec "${project_root}/../.venv/bin/python" "${project_root}/flow.py" audit \
  --flow-checkpoint "${run_root}/best.pt" \
  --output "${run_root}/audit.json" \
  --examples 4096 \
  --nearest-examples 1024 \
  --projections 128 \
  --solver-steps 4 8 16 32 \
  --c2st-epochs 100 \
  --c2st-hidden-dim 128 \
  --seed 1 \
  --device cuda
