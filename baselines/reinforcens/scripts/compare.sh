#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
../../.venv/bin/python compare.py \
  --device auto \
  --output runs/pretrained_weight_parity.json
