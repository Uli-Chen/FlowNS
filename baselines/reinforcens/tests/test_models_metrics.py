from __future__ import annotations

import numpy as np
import torch

from reinforcens.checkpoint import load_legacy_weights
from reinforcens.compare import numpy_legacy_scores
from reinforcens.metrics import list_metrics
from reinforcens.models import (
    BarycentricGenerator,
    GMF,
    MLP,
    exposure_calibrated_hard_tilt,
)
from reinforcens.trainer import negative_mmd_reward


def test_gmf_matches_legacy_numpy() -> None:
    rng = np.random.default_rng(3)
    values = [
        rng.normal(size=(4, 3)).astype(np.float32),
        rng.normal(size=(7, 3)).astype(np.float32),
        rng.normal(size=(3, 1)).astype(np.float32),
    ]
    users = np.array([0, 3], dtype=np.int32)
    items = np.array([[1, 5, 6], [2, 0, 4]], dtype=np.int32)
    model = GMF(4, 7, 3)
    load_legacy_weights(model, values)
    actual = model.score_candidates(
        torch.from_numpy(users).long(), torch.from_numpy(items).long()
    )
    expected = numpy_legacy_scores(values, users, items)
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-6, atol=1e-6)


def test_barycentric_generator_averages_policy_logits() -> None:
    torch.manual_seed(13)
    policy = GMF(4, 7, 3)
    exposure_prior = GMF(4, 7, 3)
    model = BarycentricGenerator(policy, exposure_prior)
    users = torch.tensor([0, 3])
    candidates = torch.tensor([[1, 5, 6], [2, 0, 4]])
    expected = 0.5 * (
        policy.score_candidates(users, candidates)
        + exposure_prior.score_candidates(users, candidates)
    )
    assert torch.equal(model.score_candidates(users, candidates), expected)
    model.set_prior_weight(0.25)
    expected = 0.75 * policy.score_candidates(
        users, candidates
    ) + 0.25 * exposure_prior.score_candidates(users, candidates)
    assert torch.equal(model.score_candidates(users, candidates), expected)


def test_exposure_calibrated_hard_tilt_is_safe_and_shift_invariant() -> None:
    scores = torch.tensor([[1.0, 2.0, 4.0], [-3.0, -3.0, -3.0]])
    density_logits = torch.tensor([[20.0, -20.0, 20.0], [1.0, 2.0, 3.0]])
    actual = exposure_calibrated_hard_tilt(scores, density_logits)
    shifted = exposure_calibrated_hard_tilt(scores + 100.0, density_logits)
    assert torch.allclose(actual, shifted, atol=1e-6)
    assert torch.equal(actual[1], torch.zeros(3))
    assert actual[0, 2] > actual[0, 1]
    assert actual[0, 0] == 0.0


def test_mlp_matches_legacy_numpy() -> None:
    rng = np.random.default_rng(5)
    values = [
        rng.normal(size=(4, 4)).astype(np.float32),
        rng.normal(size=(7, 4)).astype(np.float32),
        rng.normal(size=(4, 1)).astype(np.float32),
        [
            rng.normal(size=(8, 4)).astype(np.float32),
            rng.normal(size=(1, 4)).astype(np.float32),
        ],
    ]
    users = np.array([1, 2], dtype=np.int32)
    items = np.array([[0, 3], [4, 6]], dtype=np.int32)
    model = MLP(4, 7, 4, layer_count=1)
    load_legacy_weights(model, values)
    actual = model.score_candidates(
        torch.from_numpy(users).long(), torch.from_numpy(items).long()
    )
    expected = numpy_legacy_scores(values, users, items)
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-6, atol=1e-6)


def test_list_metrics_and_mmd() -> None:
    scores = np.array([[0.9, 0.1, 0.8, 0.2], [0.1, 0.9, 0.2, 0.8]])
    labels = np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=bool)
    auc, ndcg = list_metrics(scores, labels)
    assert auc[0] == 1.0 and ndcg[0] == 1.0
    assert auc[1] == 0.0 and 0.0 < ndcg[1] < 1.0

    features = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    global_reward, sample_reward = negative_mmd_reward(features, features, (1.0, 2.0))
    assert torch.allclose(global_reward, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(sample_reward, torch.zeros(2), atol=1e-6)
