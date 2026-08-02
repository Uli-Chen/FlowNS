#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reinforcens.checkpoint import load_checkpoint
from reinforcens.cli import DEFAULT_DATA, REPOSITORY_ROOT
from reinforcens.data import EvalCandidates, InteractionData
from reinforcens.metrics import list_metrics
from reinforcens.models import Recommender
from reinforcens.statistics import paired_user_bootstrap
from reinforcens.trainer import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired user bootstrap for two frozen list-evaluation checkpoints"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--cache", type=Path, default=REPOSITORY_ROOT / ".cache" / "zhihu.npz"
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--list-length", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=1, help="evaluation-list seed")
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_802)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def per_user_metrics(
    model: Recommender,
    candidates: EvalCandidates,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    auc_values: list[np.ndarray] = []
    ndcg_values: list[np.ndarray] = []
    for begin in range(0, candidates.users.size, batch_size):
        end = min(begin + batch_size, candidates.users.size)
        users = torch.from_numpy(
            candidates.users[begin:end].astype(np.int64, copy=False)
        ).to(device)
        items = torch.from_numpy(
            candidates.items[begin:end].astype(np.int64, copy=False)
        ).to(device)
        scores = model.score_candidates(users, items).float().cpu().numpy()
        auc, ndcg = list_metrics(scores, candidates.labels[begin:end])
        auc_values.append(auc)
        ndcg_values.append(ndcg)
    return np.concatenate(auc_values), np.concatenate(ndcg_values)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.resamples <= 0:
        raise SystemExit("batch size and resamples must be positive")
    device = resolve_device(args.device)
    data = InteractionData.load(
        args.data,
        num_users=16_015,
        num_items=45_782,
        cache_path=args.cache,
    )
    candidates = data.make_eval_candidates(args.split, args.list_length, args.seed)
    baseline_config, baseline, _, _ = load_checkpoint(args.baseline, device=device)
    candidate_config, candidate, _, _ = load_checkpoint(args.candidate, device=device)
    expected_shape = (data.num_users, data.num_items)
    for label, model in (("baseline", baseline), ("candidate", candidate)):
        if (model.num_users, model.num_items) != expected_shape:
            raise SystemExit(f"{label} checkpoint dimensions do not match the data")

    baseline_auc, baseline_ndcg = per_user_metrics(
        baseline, candidates, device=device, batch_size=args.batch_size
    )
    candidate_auc, candidate_ndcg = per_user_metrics(
        candidate, candidates, device=device, batch_size=args.batch_size
    )
    result = {
        "protocol": {
            "split": args.split,
            "list_length": args.list_length,
            "evaluation_seed": args.seed,
            "paired_unit": "user",
            "confidence": 0.95,
        },
        "baseline": {
            "checkpoint": str(args.baseline.resolve()),
            "model": baseline_config.model,
        },
        "candidate": {
            "checkpoint": str(args.candidate.resolve()),
            "model": candidate_config.model,
        },
        "auc": paired_user_bootstrap(
            baseline_auc,
            candidate_auc,
            resamples=args.resamples,
            seed=args.bootstrap_seed,
        ).to_dict(),
        "ndcg": paired_user_bootstrap(
            baseline_ndcg,
            candidate_ndcg,
            resamples=args.resamples,
            seed=args.bootstrap_seed + 1,
        ).to_dict(),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
