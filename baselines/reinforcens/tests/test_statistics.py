from __future__ import annotations

import numpy as np
import pytest

from reinforcens.statistics import paired_user_bootstrap


def test_paired_bootstrap_detects_uniform_improvement() -> None:
    baseline = np.linspace(0.1, 0.8, 100)
    candidate = baseline + 0.02
    result = paired_user_bootstrap(
        baseline, candidate, resamples=500, seed=7, chunk_size=31
    )
    assert result.mean_delta == pytest.approx(0.02)
    assert result.ci_lower == pytest.approx(0.02)
    assert result.ci_upper == pytest.approx(0.02)
    assert result.nonpositive_tail_probability == pytest.approx(1.0 / 501.0)
    assert result.candidate_wins == 100 and result.ties == 0


def test_paired_bootstrap_validates_pairing() -> None:
    with pytest.raises(ValueError, match="same shape"):
        paired_user_bootstrap(np.ones(3), np.ones(4), resamples=10)
    with pytest.raises(ValueError, match="finite"):
        paired_user_bootstrap(np.array([0.0, np.nan]), np.ones(2), resamples=10)
