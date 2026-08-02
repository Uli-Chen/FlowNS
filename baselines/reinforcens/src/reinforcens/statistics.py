from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    ci_lower: float
    ci_upper: float
    nonpositive_tail_probability: float
    users: int
    candidate_wins: int
    ties: int
    resamples: int
    confidence: float
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def paired_user_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20_260_802,
    chunk_size: int = 256,
) -> PairedBootstrapResult:
    """Paired percentile bootstrap over users for ``candidate - baseline``.

    Both vectors must contain metrics for the exact same user rows.  The
    reported tail probability is the add-one-smoothed fraction of bootstrap
    mean deltas that are non-positive; it is not mislabeled as a parametric
    hypothesis-test p-value.
    """

    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if baseline.shape != candidate.shape:
        raise ValueError("paired metric vectors must have the same shape")
    if baseline.size < 2:
        raise ValueError("paired bootstrap requires at least two users")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("paired metric vectors must be finite")
    if resamples <= 0 or chunk_size <= 0:
        raise ValueError("resamples and chunk_size must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    users = differences.size
    for begin in range(0, resamples, chunk_size):
        end = min(begin + chunk_size, resamples)
        indices = rng.integers(0, users, size=(end - begin, users))
        bootstrap_means[begin:end] = differences[indices].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, (tail, 1.0 - tail))
    nonpositive = int(np.count_nonzero(bootstrap_means <= 0.0))
    return PairedBootstrapResult(
        baseline_mean=float(baseline.mean()),
        candidate_mean=float(candidate.mean()),
        mean_delta=float(differences.mean()),
        ci_lower=float(lower),
        ci_upper=float(upper),
        nonpositive_tail_probability=(nonpositive + 1.0) / (resamples + 1.0),
        users=users,
        candidate_wins=int(np.count_nonzero(differences > 0.0)),
        ties=int(np.count_nonzero(differences == 0.0)),
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
