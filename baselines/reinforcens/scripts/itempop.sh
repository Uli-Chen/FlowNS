#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
../../.venv/bin/python train.py train \
  --model itempop \
  --epochs 0 \
  --eval-mode list \
  --list-length 160 \
  --device auto \
  --output runs/itempop
