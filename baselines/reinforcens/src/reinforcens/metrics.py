from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .data import EvalCandidates, InteractionData
from .models import Recommender


@dataclass(slots=True)
class EvalResult:
    primary_name: str
    primary: float
    ndcg: float
    users: int
    seconds: float
    examples_per_second: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "primary_name": self.primary_name,
            self.primary_name.lower(): self.primary,
            "ndcg": self.ndcg,
            "users": self.users,
            "seconds": self.seconds,
            "examples_per_second": self.examples_per_second,
        }


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1], dtype=order.dtype)
    return ranks


def list_metrics(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-user AUC/NDCG equivalent to ``EvaluateUser._eval_users_list``."""

    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=bool)
    if scores.shape != labels.shape or scores.ndim != 2:
        raise ValueError("scores and labels must be equally shaped matrices")
    ranks = ranks_from_scores(scores)
    batch, length = scores.shape
    positives = labels.sum(axis=1)
    negatives = length - positives
    if np.any(positives == 0) or np.any(negatives == 0):
        raise ValueError(
            "AUC/NDCG require at least one positive and one negative per user"
        )

    gains = np.where(labels, 1.0 / np.log2(ranks + 2.0), 0.0).sum(axis=1)
    discounts = 1.0 / np.log2(np.arange(length, dtype=np.float64) + 2.0)
    ideal = np.asarray([discounts[: int(count)].sum() for count in positives])
    ndcg = gains / ideal

    # Sum of global positive ranks minus their ranks among positives equals
    # the number of negative-positive inversions (the legacy anti_auc).
    positive_rank_sum = np.where(labels, ranks, 0).sum(axis=1)
    internal_rank_sum = positives * (positives - 1) / 2
    anti_auc = positive_rank_sum - internal_rank_sum
    auc = 1.0 - anti_auc / (positives * negatives)
    return auc.astype(np.float64), ndcg.astype(np.float64)


@torch.inference_mode()
def evaluate_list(
    model: Recommender,
    candidates: EvalCandidates,
    *,
    device: torch.device,
    batch_size: int = 1_024,
) -> EvalResult:
    model.eval()
    start = time.perf_counter()
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
    seconds = time.perf_counter() - start
    all_auc = np.concatenate(auc_values)
    all_ndcg = np.concatenate(ndcg_values)
    return EvalResult(
        primary_name="AUC",
        primary=float(all_auc.mean()),
        ndcg=float(all_ndcg.mean()),
        users=int(candidates.users.size),
        seconds=seconds,
        examples_per_second=float(candidates.items.size / max(seconds, 1e-12)),
    )


@torch.inference_mode()
def evaluate_topk(
    model: Recommender,
    data: InteractionData,
    *,
    split: str,
    k: int,
    device: torch.device,
    batch_size: int = 128,
) -> EvalResult:
    """Legacy all-item HR/NDCG; training positives are intentionally not masked."""

    model.eval()
    split_data = data.validation if split == "validation" else data.test
    users_np = split_data.clicks.users_with_items()
    hits: list[float] = []
    ndcgs: list[float] = []
    start = time.perf_counter()
    for begin in range(0, users_np.size, batch_size):
        batch_users_np = users_np[begin : begin + batch_size]
        users = torch.from_numpy(batch_users_np.astype(np.int64, copy=False)).to(device)
        scores = model.all_scores(users).float().cpu().numpy()
        for row, user_np in enumerate(batch_users_np):
            positives = split_data.clicks.for_user(int(user_np)).astype(
                np.int64, copy=False
            )
            positive_scores = scores[row, positives]
            # The legacy implementation counts ties as ahead, excluding the item itself.
            descending = np.sort(scores[row])[::-1]
            ranks = np.searchsorted(-descending, -positive_scores, side="right") - 1
            inside = ranks < k
            hits.append(float(inside.mean()))
            dcg = np.where(inside, 1.0 / np.log2(ranks + 2.0), 0.0).sum()
            ideal = (1.0 / np.log2(np.arange(positives.size) + 2.0)).sum()
            ndcgs.append(float(dcg / ideal))
    seconds = time.perf_counter() - start
    return EvalResult(
        primary_name="HR",
        primary=float(np.mean(hits)),
        ndcg=float(np.mean(ndcgs)),
        users=int(users_np.size),
        seconds=seconds,
        examples_per_second=float(users_np.size * data.num_items / max(seconds, 1e-12)),
    )


def format_result(result: EvalResult) -> str:
    return (
        f"{result.primary_name}={result.primary:.6f} NDCG={result.ndcg:.6f} "
        f"users={result.users} time={result.seconds:.3f}s "
        f"throughput={result.examples_per_second:,.0f} scores/s"
    )
