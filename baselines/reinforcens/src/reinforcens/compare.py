from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .checkpoint import (
    infer_legacy_architecture,
    load_checkpoint,
    load_legacy_weights,
    read_legacy_pickle,
)
from .cli import BASELINE_ROOT, DEFAULT_DATA, DEFAULT_PRETRAINED
from .data import EvalCandidates, InteractionData
from .metrics import list_metrics
from .models import Recommender, build_model
from .trainer import resolve_device


def numpy_legacy_scores(
    values: list[Any], users: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    architecture, _, _, _, _ = infer_legacy_architecture(values)
    user_embedding = np.asarray(values[0], dtype=np.float32)[users]
    item_embedding = np.asarray(values[1], dtype=np.float32)[candidates]
    h = np.asarray(values[2], dtype=np.float32).reshape(-1)
    if architecture == "gmf":
        return ((user_embedding[:, None, :] * item_embedding) * h).sum(axis=2)
    feature = np.concatenate(
        (
            np.broadcast_to(user_embedding[:, None, :], item_embedding.shape),
            item_embedding,
        ),
        axis=2,
    )
    for weight_bias in values[3:]:
        weight, bias = (np.asarray(value, dtype=np.float32) for value in weight_bias)
        feature = np.maximum(feature @ weight + bias, 0.0)
    return (feature * h).sum(axis=2)


def score_numpy(
    values: list[Any], candidates: EvalCandidates, batch_size: int
) -> tuple[np.ndarray, float]:
    output = np.empty(candidates.items.shape, dtype=np.float32)
    start = time.perf_counter()
    for begin in range(0, candidates.users.size, batch_size):
        end = min(begin + batch_size, candidates.users.size)
        output[begin:end] = numpy_legacy_scores(
            values, candidates.users[begin:end], candidates.items[begin:end]
        )
    return output, time.perf_counter() - start


@torch.inference_mode()
def score_torch(
    model: Recommender,
    candidates: EvalCandidates,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    output = np.empty(candidates.items.shape, dtype=np.float32)
    model.eval()
    # Warm up kernels without including setup in the benchmark.
    warm_end = min(batch_size, candidates.users.size)
    warm_users = torch.from_numpy(candidates.users[:warm_end].astype(np.int64)).to(
        device
    )
    warm_items = torch.from_numpy(candidates.items[:warm_end].astype(np.int64)).to(
        device
    )
    model.score_candidates(warm_users, warm_items)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for begin in range(0, candidates.users.size, batch_size):
        end = min(begin + batch_size, candidates.users.size)
        users = torch.from_numpy(
            candidates.users[begin:end].astype(np.int64, copy=False)
        ).to(device)
        items = torch.from_numpy(
            candidates.items[begin:end].astype(np.int64, copy=False)
        ).to(device)
        output[begin:end] = model.score_candidates(users, items).float().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, time.perf_counter() - start


def summarize_backend(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    seconds: float,
    device: str,
) -> dict[str, Any]:
    auc, ndcg = list_metrics(scores, labels)
    return {
        "backend": name,
        "device": device,
        "auc": float(auc.mean()),
        "ndcg": float(ndcg.mean()),
        "seconds": seconds,
        "scores_per_second": float(scores.size / max(seconds, 1e-12)),
    }


def compare_scores(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    reference_order = np.argsort(-reference, axis=1, kind="stable")
    candidate_order = np.argsort(-candidate, axis=1, kind="stable")
    return {
        "max_absolute_score_error": float(difference.max()),
        "mean_absolute_score_error": float(difference.mean()),
        "identical_full_rank_rows": int(
            np.all(reference_order == candidate_order, axis=1).sum()
        ),
        "total_rank_rows": int(reference.shape[0]),
        "top1_agreement": float(
            np.mean(reference_order[:, 0] == candidate_order[:, 0])
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the released BPR-GMF weights against PyTorch RNS"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--cache", type=Path, default=BASELINE_ROOT / ".cache" / "zhihu.npz"
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        "--legacy-checkpoint",
        dest="pretrained_checkpoint",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument(
        "--pytorch-checkpoint",
        type=Path,
        action="append",
        help="Also evaluate a trained .pt model; repeat for several checkpoints",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--list-length", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--num-users", type=int, default=16_015)
    parser.add_argument("--num-items", type=int, default=45_782)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-exposure-conflicts", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data = InteractionData.load(
        args.data,
        num_users=args.num_users,
        num_items=args.num_items,
        clean_exposure_conflicts=not args.keep_exposure_conflicts,
        cache_path=args.cache,
    )
    candidates = data.make_eval_candidates(args.split, args.list_length, args.seed)
    values = read_legacy_pickle(args.pretrained_checkpoint)
    architecture, num_users, num_items, embedding_dim, layer_count = (
        infer_legacy_architecture(values)
    )
    model = build_model(architecture, num_users, num_items, embedding_dim, layer_count)
    load_legacy_weights(model, values)
    device = resolve_device(args.device)
    model.to(device)

    numpy_scores, numpy_seconds = score_numpy(values, candidates, args.batch_size)
    torch_scores, torch_seconds = score_torch(
        model, candidates, device, args.batch_size
    )
    result: dict[str, Any] = {
        "dataset": data.statistics(),
        "split": args.split,
        "candidate_shape": list(candidates.items.shape),
        "pretrained_checkpoint": str(args.pretrained_checkpoint.resolve()),
        "torch_version": torch.__version__,
        "device": str(device),
        "backends": [
            summarize_backend(
                "legacy_numpy_equations",
                numpy_scores,
                candidates.labels,
                numpy_seconds,
                "cpu",
            ),
            summarize_backend(
                "pytorch_converted_legacy_weights",
                torch_scores,
                candidates.labels,
                torch_seconds,
                str(device),
            ),
        ],
        "pytorch_vs_legacy": compare_scores(numpy_scores, torch_scores),
    }
    for checkpoint in args.pytorch_checkpoint or []:
        trained_config, trained, _, metadata = load_checkpoint(
            checkpoint, device=device
        )
        trained_scores, trained_seconds = score_torch(
            trained, candidates, device, args.batch_size
        )
        trained_summary = summarize_backend(
            f"pytorch_trained_{trained_config.model}",
            trained_scores,
            candidates.labels,
            trained_seconds,
            str(device),
        )
        trained_summary["checkpoint"] = str(checkpoint.resolve())
        trained_summary["training_config"] = trained_config.to_dict()
        trained_summary["metadata"] = metadata
        result["backends"].append(trained_summary)

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
