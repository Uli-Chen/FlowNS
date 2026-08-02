from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from reinforcens.data import InteractionData

from .flow import ConditionalExposureFlow


@dataclass(slots=True)
class ExposureFlowAuditConfig:
    examples: int = 4_096
    nearest_examples: int = 1_024
    projections: int = 128
    solver_steps: tuple[int, ...] = (4, 8, 16, 32)
    c2st_epochs: int = 100
    c2st_hidden_dim: int = 128
    seed: int = 1

    def __post_init__(self) -> None:
        if min(
            self.examples,
            self.nearest_examples,
            self.projections,
            self.c2st_epochs,
            self.c2st_hidden_dim,
        ) <= 0:
            raise ValueError("flow audit counts must be positive")
        if not self.solver_steps or min(self.solver_steps) <= 0:
            raise ValueError("solver steps must be positive")
        if len(set(self.solver_steps)) != len(self.solver_steps):
            raise ValueError("solver steps must be unique")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["solver_steps"] = list(self.solver_steps)
        return values


def _device_of(flow: ConditionalExposureFlow) -> torch.device:
    return flow.catalog_mean.device


@torch.inference_mode()
def _nearest_catalog_metrics(
    data: InteractionData,
    flow: ConditionalExposureFlow,
    users: Tensor,
    generated: Tensor,
    catalog_embeddings: Tensor,
    count: int,
) -> dict[str, float | int]:
    count = min(count, generated.shape[0])
    generated = generated[:count].float()
    users_np = users[:count].cpu().numpy().astype(np.int32, copy=False)
    catalog = flow.normalize(catalog_embeddings).float()
    distances_out: list[Tensor] = []
    items_out: list[Tensor] = []
    for begin in range(0, count, 128):
        distances = torch.cdist(generated[begin : begin + 128], catalog)
        values, items = distances.min(dim=1)
        distances_out.append(values.cpu())
        items_out.append(items.cpu())
    nearest = torch.cat(items_out).numpy().astype(np.int32, copy=False)
    distances_np = torch.cat(distances_out).numpy()
    return {
        "nearest_examples": count,
        "nearest_catalog_distance_mean": float(distances_np.mean()),
        "nearest_catalog_distance_p95": float(np.quantile(distances_np, 0.95)),
        "nearest_train_click_rate": float(
            data.train_clicks.contains(users_np, nearest).mean()
        ),
        "nearest_train_exposure_rate": float(
            data.train_exposures.contains(users_np, nearest).mean()
        ),
        "nearest_unique_item_ratio": float(np.unique(nearest).size / count),
    }


def _conditional_c2st(
    flow: ConditionalExposureFlow,
    users: Tensor,
    generated: Tensor,
    target: Tensor,
    config: ExposureFlowAuditConfig,
) -> dict[str, float | int]:
    """Train-only classifier two-sample test conditioned on flow user features."""

    device = generated.device
    with torch.no_grad():
        condition = flow.user_embedding(users).detach().float()
        features = torch.cat(
            (
                torch.cat((generated.float(), condition), dim=1),
                torch.cat((target.float(), condition), dim=1),
            ),
            dim=0,
        )
        labels = torch.cat(
            (
                torch.zeros(users.numel(), device=device),
                torch.ones(users.numel(), device=device),
            )
        )
    generator = torch.Generator(device=device).manual_seed(config.seed + 50_000)
    pair_order = torch.randperm(users.numel(), generator=generator, device=device)
    train_pairs_count = max(1, int(0.6 * pair_order.numel()))
    validation_pairs_count = max(1, int(0.2 * pair_order.numel()))
    if train_pairs_count + validation_pairs_count >= pair_order.numel():
        train_pairs_count = pair_order.numel() - 2
        validation_pairs_count = 1
    train_pairs = pair_order[:train_pairs_count]
    validation_pairs = pair_order[
        train_pairs_count : train_pairs_count + validation_pairs_count
    ]
    test_pairs = pair_order[train_pairs_count + validation_pairs_count :]

    def balanced_indices(pairs: Tensor) -> Tensor:
        return torch.cat((pairs, pairs + users.numel()))

    train_indices = balanced_indices(train_pairs)
    validation_indices = balanced_indices(validation_pairs)
    test_indices = balanced_indices(test_pairs)
    mean = features[train_indices].mean(dim=0)
    scale = features[train_indices].std(dim=0, unbiased=False).clamp_min(1e-5)
    features = (features - mean) / scale
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 50_003)
        classifier = nn.Sequential(
            nn.Linear(features.shape[1], config.c2st_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.c2st_hidden_dim, 1),
        ).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=3e-3, weight_decay=1e-3)
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, config.c2st_epochs + 1):
        classifier.train()
        logits = classifier(features[train_indices]).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels[train_indices]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        classifier.eval()
        with torch.inference_mode():
            validation_logits = classifier(features[validation_indices]).squeeze(1)
            validation_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                validation_logits, labels[validation_indices]
            )
        if float(validation_loss) < best_validation_loss:
            best_validation_loss = float(validation_loss)
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in classifier.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("conditional C2ST produced no selected classifier")
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.inference_mode():
        train_predictions = classifier(features[train_indices]).squeeze(1) >= 0
        validation_predictions = (
            classifier(features[validation_indices]).squeeze(1) >= 0
        )
        test_predictions = classifier(features[test_indices]).squeeze(1) >= 0
        train_accuracy = (
            train_predictions == labels[train_indices].bool()
        ).float().mean()
        test_accuracy = (
            test_predictions == labels[test_indices].bool()
        ).float().mean()
        validation_accuracy = (
            validation_predictions == labels[validation_indices].bool()
        ).float().mean()
    return {
        "conditional_c2st_train_examples": int(train_indices.numel()),
        "conditional_c2st_validation_examples": int(validation_indices.numel()),
        "conditional_c2st_test_examples": int(test_indices.numel()),
        "conditional_c2st_selected_epoch": best_epoch,
        "conditional_c2st_train_accuracy": float(train_accuracy),
        "conditional_c2st_validation_accuracy": float(validation_accuracy),
        "conditional_c2st_test_accuracy": float(test_accuracy),
    }


def audit_exposure_flow(
    data: InteractionData,
    flow: ConditionalExposureFlow,
    catalog_embeddings: Tensor,
    exposure_holdout_keys: np.ndarray,
    config: ExposureFlowAuditConfig,
) -> dict[str, Any]:
    """Audit solver convergence and distribution fit using train-only exposures."""

    if catalog_embeddings.shape != (data.num_items, flow.config.embedding_dim):
        raise ValueError("catalog embedding shape does not match data and flow")
    keys = np.asarray(exposure_holdout_keys, dtype=np.int64)
    if keys.size < 2:
        raise ValueError("flow audit requires at least two holdout exposures")
    count = min(config.examples, keys.size)
    if count < 3:
        raise ValueError("flow audit requires at least three C2ST pairs")
    selected = np.random.default_rng(config.seed + 40_000).choice(
        keys.size, size=count, replace=False
    )
    selected_keys = keys[selected]
    device = _device_of(flow)
    users = torch.from_numpy(
        (selected_keys // data.num_items).astype(np.int64, copy=False)
    ).to(device)
    items = torch.from_numpy(
        (selected_keys % data.num_items).astype(np.int64, copy=False)
    ).to(device)
    catalog = catalog_embeddings.detach().float().to(device)
    target = flow.normalize(catalog[items]).float()
    generator = torch.Generator(device=device).manual_seed(config.seed + 40_003)
    noise = torch.randn(
        count, flow.config.embedding_dim, generator=generator, device=device
    )
    flow.eval()
    endpoints: dict[int, Tensor] = {}
    with torch.inference_mode():
        for steps in sorted(config.solver_steps):
            endpoints[steps] = flow.integrate(
                users, noise, steps=steps, solver="heun"
            ).float()

    projection_generator = torch.Generator(device=device).manual_seed(
        config.seed + 40_007
    )
    directions = torch.randn(
        config.projections,
        flow.config.embedding_dim,
        generator=projection_generator,
        device=device,
    )
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    target_projection = torch.sort(target @ directions.T, dim=0).values
    reference_steps = max(config.solver_steps)
    reference = endpoints[reference_steps]
    result: dict[str, Any] = {
        "config": config.to_dict(),
        "train_split_holdout_only": True,
        "official_validation_or_test_used": False,
        "examples": count,
        "reference_steps": reference_steps,
        "target_norm_mean": float(target.norm(dim=1).mean()),
        "generated_norm_mean": float(reference.norm(dim=1).mean()),
    }
    for steps in sorted(config.solver_steps):
        endpoint = endpoints[steps]
        projection = torch.sort(endpoint @ directions.T, dim=0).values
        swd = (projection - target_projection).square().mean().sqrt()
        result[f"steps_{steps}_target_sliced_wasserstein"] = float(swd)
        result[f"steps_{steps}_to_reference_rmse"] = float(
            (endpoint - reference).square().mean().sqrt()
        )
    result.update(
        _nearest_catalog_metrics(
            data,
            flow,
            users,
            reference,
            catalog,
            config.nearest_examples,
        )
    )
    result.update(_conditional_c2st(flow, users, reference, target, config))
    return result
