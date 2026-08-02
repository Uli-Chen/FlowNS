from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from reinforcens.data import InteractionData
from flowrns.flow import (
    ConditionalExposureFlow,
    ExposureFlowConfig,
    sinusoidal_time_embedding,
)
from flowrns.flow_audit import ExposureFlowAuditConfig, audit_exposure_flow
from flowrns.flow_ranker import (
    ContinuousFlowRankerTrainer,
    ContinuousRankerConfig,
    hard_bpr_loss,
)
from flowrns.flow_training import (
    ExposureFlowTrainer,
    FlowTrainingConfig,
    load_exposure_flow,
    sliced_wasserstein_distance,
    split_train_exposures,
)
from reinforcens.models import GMF
from flow_test_data import write_tiny_dataset


def tiny_flow() -> ConditionalExposureFlow:
    torch.manual_seed(3)
    catalog = torch.randn(11, 4)
    config = ExposureFlowConfig(
        num_users=5,
        embedding_dim=4,
        user_dim=6,
        time_dim=8,
        hidden_dim=16,
        depth=2,
        sampling_steps=4,
    )
    return ConditionalExposureFlow(config, catalog)


def test_time_embedding_shape_and_validation() -> None:
    values = sinusoidal_time_embedding(torch.tensor([0.0, 0.5, 1.0]), 8)
    assert values.shape == (3, 8)
    assert torch.isfinite(values).all()
    with pytest.raises(ValueError):
        sinusoidal_time_embedding(torch.tensor([0.5]), 7)


def test_zero_velocity_flow_preserves_noise() -> None:
    flow = tiny_flow()
    for parameter in flow.parameters():
        parameter.data.zero_()
    users = torch.tensor([0, 3])
    noise = torch.randn(2, 4)
    endpoint = flow.integrate(users, noise, steps=8, solver="heun")
    torch.testing.assert_close(endpoint, noise)


def test_flow_matching_loss_has_finite_gradients() -> None:
    flow = tiny_flow()
    users = torch.tensor([0, 1, 2])
    targets = torch.randn(3, 4)
    loss, metrics = flow.flow_matching_loss(
        users,
        targets,
        times=torch.tensor([0.1, 0.5, 0.9]),
        noise=torch.zeros_like(targets),
    )
    loss.backward()
    assert loss.item() > 0
    assert set(metrics) == {"velocity_mse", "endpoint_mse"}
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in flow.parameters()
    )


def test_gmf_continuous_score_matches_item_lookup() -> None:
    torch.manual_seed(7)
    model = GMF(num_users=4, num_items=9, embedding_dim=5)
    users = torch.tensor([0, 2, 3])
    items = torch.tensor([1, 5, 8])
    expected = model.score(users, items)
    actual = model.score_embeddings(users, model.item_embedding(items))
    torch.testing.assert_close(actual, expected)
    expected_candidates = model.score_candidates(
        users, torch.stack((items, items), dim=1)
    )
    candidate_embeddings = model.item_embedding(
        torch.stack((items, items), dim=1)
    )
    actual_candidates = model.score_embedding_candidates(
        users, candidate_embeddings
    )
    torch.testing.assert_close(actual_candidates, expected_candidates)


def test_gmf_continuous_score_rejects_wrong_shape() -> None:
    model = GMF(num_users=2, num_items=3, embedding_dim=4)
    with pytest.raises(ValueError):
        model.score_embeddings(torch.tensor([0, 1]), torch.zeros(2, 3))


def test_train_exposure_split_is_disjoint_and_reproducible() -> None:
    keys = torch.arange(100).numpy()
    first_train, first_validation = split_train_exposures(
        keys,
        validation_fraction=0.2,
        validation_examples=50,
        seed=11,
    )
    second_train, second_validation = split_train_exposures(
        keys,
        validation_fraction=0.2,
        validation_examples=50,
        seed=11,
    )
    assert len(first_validation) == 20
    assert not set(first_train) & set(first_validation)
    assert set(first_train) | set(first_validation) == set(keys)
    assert (first_train == second_train).all()
    assert (first_validation == second_validation).all()


def test_sliced_wasserstein_is_zero_for_equal_samples() -> None:
    samples = torch.randn(32, 4)
    generator = torch.Generator().manual_seed(1)
    distance = sliced_wasserstein_distance(
        samples, samples.clone(), projections=8, generator=generator
    )
    assert distance.item() == pytest.approx(0.0, abs=1e-7)


def test_flow_checkpoint_round_trip(tmp_path) -> None:
    flow = tiny_flow()
    path = tmp_path / "flow.pt"
    torch.save(
        {
            "format_version": 1,
            "flow_config": flow.config.to_dict(),
            "epoch": 2,
            "metrics": {"validation_loss": 1.0},
            "history": [],
            "training_exposures": 10,
            "validation_exposures": 2,
            "flow": flow.state_dict(),
        },
        path,
    )
    catalog = torch.randn(11, 4)
    loaded, metadata = load_exposure_flow(path, catalog)
    assert loaded.config == flow.config
    assert metadata["epoch"] == 2
    for expected, actual in zip(flow.parameters(), loaded.parameters()):
        torch.testing.assert_close(actual, expected)


def test_continuous_ranker_keeps_catalog_fixed(tmp_path: Path) -> None:
    data = InteractionData.load(
        write_tiny_dataset(tmp_path / "data"), num_users=3, num_items=10
    )
    ranker = GMF(3, 10, 4)
    flow = ConditionalExposureFlow(
        ExposureFlowConfig(
            num_users=3,
            embedding_dim=4,
            user_dim=4,
            time_dim=4,
            hidden_dim=8,
            depth=1,
            sampling_steps=2,
        ),
        ranker.item_embedding.weight.detach(),
    )
    config = ContinuousRankerConfig(
        batch_size=2,
        epochs=1,
        minimum_epochs=1,
        early_stopping_patience=1,
        candidates_per_user=2,
        pool_batch_size=3,
        pool_refresh_epochs=1,
        flow_steps=2,
        recommendation_list_length=5,
        eval_batch_size=2,
        maximum_train_examples=4,
        diagnostics_examples=3,
        device="cpu",
    )
    trainer = ContinuousFlowRankerTrainer(data, ranker, flow, config)
    catalog_before = ranker.item_embedding.weight.detach().clone()
    user_before = ranker.user_embedding.weight.detach().clone()
    history = trainer.fit(tmp_path / "run")
    assert len(history) == 2
    assert torch.equal(catalog_before, ranker.item_embedding.weight)
    assert not torch.equal(user_before, ranker.user_embedding.weight)
    assert (tmp_path / "run" / "best.pt").is_file()
    assert (tmp_path / "run" / "summary.json").is_file()
    with pytest.raises(ValueError, match="direct continuous"):
        ContinuousRankerConfig(
            negative_source="flow", freeze_item_embeddings=False
        )


def test_flow_audit_uses_train_holdout_and_reports_solver_gap(tmp_path: Path) -> None:
    data = InteractionData.load(
        write_tiny_dataset(tmp_path / "audit-data"), num_users=3, num_items=10
    )
    catalog = torch.randn(10, 4)
    flow = ConditionalExposureFlow(
        ExposureFlowConfig(
            num_users=3,
            embedding_dim=4,
            user_dim=4,
            time_dim=4,
            hidden_dim=8,
            depth=1,
            sampling_steps=2,
        ),
        catalog,
    )
    result = audit_exposure_flow(
        data,
        flow,
        catalog,
        data.train_exposures.keys[:3],
        ExposureFlowAuditConfig(
            examples=3,
            nearest_examples=2,
            projections=2,
            solver_steps=(1, 2),
            c2st_epochs=2,
            c2st_hidden_dim=4,
            seed=3,
        ),
    )
    assert result["official_validation_or_test_used"] is False
    assert result["steps_2_to_reference_rmse"] == pytest.approx(0.0)
    assert 0.0 <= result["conditional_c2st_test_accuracy"] <= 1.0


def test_safe_hard_continuous_selection_respects_ess_floor(tmp_path: Path) -> None:
    data = InteractionData.load(
        write_tiny_dataset(tmp_path / "safe-hard-data"), num_users=3, num_items=10
    )
    ranker = GMF(3, 10, 4)
    flow = ConditionalExposureFlow(
        ExposureFlowConfig(
            num_users=3,
            embedding_dim=4,
            user_dim=4,
            time_dim=4,
            hidden_dim=8,
            depth=1,
            sampling_steps=1,
        ),
        ranker.item_embedding.weight.detach(),
    )
    trainer = ContinuousFlowRankerTrainer(
        data,
        ranker,
        flow,
        ContinuousRankerConfig(
            epochs=1,
            minimum_epochs=1,
            early_stopping_patience=1,
            candidates_per_user=4,
            pool_batch_size=4,
            flow_steps=1,
            selection_mode="safe-hard",
            minimum_ess_ratio=0.75,
            recommendation_list_length=5,
            diagnostics_examples=2,
            device="cpu",
        ),
    )
    trainer.refresh_negative_pool(epoch=1)
    users_np = torch.tensor([0, 1, 2], dtype=torch.int32).numpy()
    users = torch.tensor([0, 1, 2])
    negatives, weights, diagnostics = trainer._negative_embeddings(users_np, users)
    assert negatives.shape == (3, 4)
    assert weights is None
    assert diagnostics["selection_ess_ratio"] >= 0.75 - 1e-4
    assert diagnostics["selection_kl_to_flow_pool"] >= 0.0


def test_soft_transport_maps_only_to_train_exposure_neighbors(tmp_path: Path) -> None:
    data = InteractionData.load(
        write_tiny_dataset(tmp_path / "transport-data"), num_users=3, num_items=10
    )
    ranker = GMF(3, 10, 4)
    flow = ConditionalExposureFlow(
        ExposureFlowConfig(
            num_users=3,
            embedding_dim=4,
            user_dim=4,
            time_dim=4,
            hidden_dim=8,
            depth=1,
            sampling_steps=1,
        ),
        ranker.item_embedding.weight.detach(),
    )
    trainer = ContinuousFlowRankerTrainer(
        data,
        ranker,
        flow,
        ContinuousRankerConfig(
            epochs=1,
            minimum_epochs=1,
            early_stopping_patience=1,
            candidates_per_user=2,
            pool_batch_size=3,
            flow_steps=1,
            negative_source="soft-flow",
            transport_neighbors=2,
            transport_minimum_ess_ratio=0.75,
            recommendation_list_length=5,
            diagnostics_examples=2,
            device="cpu",
        ),
    )
    diagnostics = trainer.refresh_negative_pool(epoch=1)
    users_np = np.asarray([0, 1, 2], dtype=np.int32)
    users = torch.tensor([0, 1, 2])
    negatives, weights, _ = trainer._negative_embeddings(users_np, users)
    assert negatives.shape == (3, 2, 4)
    assert weights is not None
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
    assert diagnostics["transport_ess_ratio"] >= 0.75 - 1e-4
    assert trainer.transport_items is not None
    for user in range(3):
        exposed = set(data.train_exposures.for_user(user).tolist())
        if exposed:
            assert set(trainer.transport_items[user].reshape(-1).tolist()) <= exposed


def test_mapped_transport_can_train_ranker_item_table(tmp_path: Path) -> None:
    data = InteractionData.load(
        write_tiny_dataset(tmp_path / "trainable-map-data"),
        num_users=3,
        num_items=10,
    )
    ranker = GMF(3, 10, 4)
    flow = ConditionalExposureFlow(
        ExposureFlowConfig(
            num_users=3,
            embedding_dim=4,
            user_dim=4,
            time_dim=4,
            hidden_dim=8,
            depth=1,
            sampling_steps=1,
        ),
        ranker.item_embedding.weight.detach(),
    )
    trainer = ContinuousFlowRankerTrainer(
        data,
        ranker,
        flow,
        ContinuousRankerConfig(
            batch_size=2,
            epochs=1,
            minimum_epochs=1,
            early_stopping_patience=1,
            candidates_per_user=2,
            pool_batch_size=3,
            flow_steps=1,
            negative_source="soft-flow",
            freeze_item_embeddings=False,
            transport_neighbors=2,
            recommendation_list_length=5,
            maximum_train_examples=4,
            diagnostics_examples=2,
            device="cpu",
        ),
    )
    item_before = ranker.item_embedding.weight.detach().clone()
    trainer.fit(tmp_path / "trainable-map-run")
    assert not torch.equal(item_before, ranker.item_embedding.weight)


def test_flow_dns_uses_bounded_mixed_candidate_pool(tmp_path: Path) -> None:
    data = InteractionData.load(
        write_tiny_dataset(tmp_path / "flow-dns-data"), num_users=3, num_items=10
    )
    ranker = GMF(3, 10, 4)
    flow = ConditionalExposureFlow(
        ExposureFlowConfig(
            num_users=3,
            embedding_dim=4,
            user_dim=4,
            time_dim=4,
            hidden_dim=8,
            depth=1,
            sampling_steps=1,
        ),
        ranker.item_embedding.weight.detach(),
    )
    trainer = ContinuousFlowRankerTrainer(
        data,
        ranker,
        flow,
        ContinuousRankerConfig(
            epochs=1,
            minimum_epochs=1,
            early_stopping_patience=1,
            candidates_per_user=2,
            pool_batch_size=3,
            flow_steps=1,
            negative_source="flow-dns",
            freeze_item_embeddings=False,
            transport_neighbors=2,
            dns_uniform_candidates=2,
            dns_flow_candidates=1,
            recommendation_list_length=5,
            diagnostics_examples=2,
            device="cpu",
        ),
    )
    trainer.refresh_negative_pool(epoch=1)
    users_np = np.asarray([0, 1, 2], dtype=np.int32)
    users = torch.tensor([0, 1, 2])
    negatives, weights, diagnostics = trainer._negative_embeddings(users_np, users)
    assert negatives.shape == (3, 4)
    assert weights is None
    assert diagnostics["dns_flow_candidate_ratio"] == pytest.approx(1 / 3)
    assert 0.0 <= diagnostics["dns_selected_flow_rate"] <= 1.0


def test_hard_bpr_contains_original_bpr_as_special_case() -> None:
    differences = torch.tensor([-3.0, 0.0, 2.0])
    expected = torch.nn.functional.softplus(-differences)
    actual = hard_bpr_loss(differences, a=0.0, b=0.0, c=1.0)
    torch.testing.assert_close(actual, expected)
    robust = hard_bpr_loss(differences, a=1.0, b=-1.0, c=0.8)
    assert torch.isfinite(robust).all()


def test_flow_warm_start_preserves_split_and_records_provenance(tmp_path: Path) -> None:
    data = InteractionData.load(
        write_tiny_dataset(tmp_path / "warm-data"), num_users=3, num_items=10
    )
    catalog = torch.randn(10, 4)
    flow_config = ExposureFlowConfig(
        num_users=3,
        embedding_dim=4,
        user_dim=4,
        time_dim=4,
        hidden_dim=8,
        depth=1,
        sampling_steps=1,
    )
    source_flow = ConditionalExposureFlow(flow_config, catalog)
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "format_version": 1,
            "flow_config": flow_config.to_dict(),
            "epoch": 7,
            "metrics": {"validation_loss": 0.5},
            "flow": source_flow.state_dict(),
        },
        checkpoint,
    )
    trainer = ExposureFlowTrainer(
        data,
        catalog,
        flow_config,
        FlowTrainingConfig(
            epochs=1,
            minimum_epochs=1,
            early_stopping_patience=1,
            batch_size=2,
            validation_examples=2,
            diagnostics_examples=2,
            diagnostics_projections=2,
            device="cpu",
        ),
    )
    training_keys = trainer.training_keys.copy()
    validation_keys = trainer.validation_keys.copy()
    provenance = trainer.initialize_from_checkpoint(checkpoint)
    assert provenance["source_epoch"] == 7
    assert provenance["optimizer_state_reused"] is False
    assert (trainer.training_keys == training_keys).all()
    assert (trainer.validation_keys == validation_keys).all()
    for expected, actual in zip(source_flow.parameters(), trainer.flow.parameters()):
        torch.testing.assert_close(expected, actual)
