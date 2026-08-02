from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from reinforcens.data import EvalCandidates, InteractionData
from reinforcens.metrics import EvalResult, evaluate_list, format_result
from reinforcens.models import GMF
from reinforcens.sampling import NegativeSampler
from reinforcens.trainer import resolve_device, seed_everything

from .flow import ConditionalExposureFlow


def hard_bpr_loss(
    score_difference: Tensor, *, a: float, b: float, c: float
) -> Tensor:
    """Hard-BPR from Shi et al. for robust dynamic hard negatives."""

    probability = (torch.sigmoid(c * score_difference + b) + a) / (1.0 + a)
    return -torch.log(probability.clamp_min(1e-12))


@dataclass(slots=True)
class ContinuousRankerConfig:
    """Train a frozen-catalog GMF with continuous flow negatives.

    Freezing the item table is essential: the exposure flow was fitted in the
    released GMF catalog coordinates. Updating that table would silently move
    positive items while leaving generated negatives in the old coordinates.
    """

    batch_size: int = 2_048
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    discriminator_reg: float = 1e-5
    epochs: int = 30
    minimum_epochs: int = 10
    early_stopping_patience: int = 6
    minimum_ndcg_improvement: float = 1e-5
    candidates_per_user: int = 16
    pool_batch_size: int = 2_048
    pool_refresh_epochs: int = 5
    flow_steps: int = 16
    flow_solver: str = "heun"
    negative_source: str = "flow"
    freeze_item_embeddings: bool = True
    selection_mode: str = "random"
    hardness_beta: float = 2.0
    minimum_ess_ratio: float = 0.5
    safety_temperature: float = 1.0
    transport_neighbors: int = 8
    transport_minimum_ess_ratio: float = 0.5
    transport_max_inverse_temperature: float = 20.0
    dns_uniform_candidates: int = 29
    dns_flow_candidates: int = 1
    ranking_loss: str = "bpr"
    hard_bpr_a: float = 1.0
    hard_bpr_b: float = -1.0
    hard_bpr_c: float = 0.8
    recommendation_list_length: int = 160
    eval_batch_size: int = 1_024
    maximum_train_examples: int | None = None
    diagnostics_examples: int = 1_024
    seed: int = 1
    device: str = "auto"
    amp: bool = False
    gradient_clip: float = 0.0

    def __post_init__(self) -> None:
        self.negative_source = self.negative_source.lower()
        self.selection_mode = self.selection_mode.lower()
        self.ranking_loss = self.ranking_loss.lower()
        self.flow_solver = self.flow_solver.lower()
        if self.negative_source not in {
            "flow",
            "soft-flow",
            "flow-dns",
            "uniform",
        }:
            raise ValueError(
                "negative_source must be flow, soft-flow, flow-dns, or uniform"
            )
        if self.flow_solver not in {"euler", "heun"}:
            raise ValueError("flow_solver must be euler or heun")
        if self.ranking_loss not in {"bpr", "hard-bpr"}:
            raise ValueError("ranking_loss must be bpr or hard-bpr")
        if self.selection_mode not in {"random", "safe-hard"}:
            raise ValueError("selection_mode must be random or safe-hard")
        if self.negative_source in {"uniform", "flow-dns"} and (
            self.selection_mode != "random"
        ):
            raise ValueError("safe-hard selection requires flow-based negatives")
        if self.negative_source == "flow" and not self.freeze_item_embeddings:
            raise ValueError("direct continuous flow requires frozen item embeddings")
        if min(
            self.batch_size,
            self.epochs,
            self.minimum_epochs,
            self.early_stopping_patience,
            self.candidates_per_user,
            self.pool_batch_size,
            self.pool_refresh_epochs,
            self.flow_steps,
            self.transport_neighbors,
            self.dns_uniform_candidates,
            self.dns_flow_candidates,
            self.recommendation_list_length,
            self.eval_batch_size,
            self.diagnostics_examples,
        ) <= 0:
            raise ValueError("continuous-ranker counts must be positive")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs cannot exceed epochs")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid continuous-ranker optimizer configuration")
        if self.discriminator_reg < 0 or self.minimum_ndcg_improvement < 0:
            raise ValueError("regularization and NDCG improvement must be non-negative")
        if self.gradient_clip < 0:
            raise ValueError("gradient_clip must be non-negative")
        if self.hardness_beta < 0 or self.safety_temperature <= 0:
            raise ValueError("hardness beta and safety temperature are invalid")
        if not 0 < self.minimum_ess_ratio <= 1:
            raise ValueError("minimum_ess_ratio must be in (0, 1]")
        if not 0 < self.transport_minimum_ess_ratio <= 1:
            raise ValueError("transport_minimum_ess_ratio must be in (0, 1]")
        if self.transport_max_inverse_temperature < 0:
            raise ValueError("transport_max_inverse_temperature must be non-negative")
        if self.hard_bpr_a < 0 or self.hard_bpr_c <= 0:
            raise ValueError("Hard-BPR requires a >= 0 and c > 0")
        if self.maximum_train_examples is not None and self.maximum_train_examples <= 0:
            raise ValueError("maximum_train_examples must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContinuousFlowRankerTrainer:
    """BPR ranker whose negative item is a continuous flow endpoint."""

    def __init__(
        self,
        data: InteractionData,
        ranker: GMF,
        flow: ConditionalExposureFlow | None,
        config: ContinuousRankerConfig,
    ) -> None:
        if (ranker.num_users, ranker.num_items) != (data.num_users, data.num_items):
            raise ValueError("ranker dimensions do not match the dataset")
        if flow is None and config.negative_source == "flow":
            raise ValueError("flow negatives require a fitted exposure flow")
        if flow is not None:
            if flow.config.num_users != data.num_users:
                raise ValueError("flow user count does not match the dataset")
            if flow.config.embedding_dim != ranker.embedding_dim:
                raise ValueError("flow and ranker embedding dimensions differ")

        self.data = data
        self.config = config
        self.device = resolve_device(config.device)
        if config.amp and self.device.type != "cuda":
            raise ValueError("continuous-ranker AMP requires CUDA")
        seed_everything(config.seed)
        self.rng = np.random.default_rng(config.seed + 401)
        self.sampler = NegativeSampler(data, config.seed + 409)
        self.ranker = ranker.to(self.device)
        self.ranker.item_embedding.weight.requires_grad_(
            not config.freeze_item_embeddings
        )
        self.flow = (
            None
            if flow is None
            else flow.to(self.device).eval().requires_grad_(False)
        )
        # Flow generation and transport always use the immutable released
        # geometry.  A mapped discrete ranker may update its own item table.
        self.catalog = self.ranker.item_embedding.weight.detach().clone()
        self.optimizer = torch.optim.Adam(
            (parameter for parameter in self.ranker.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
        self.eval_candidates: EvalCandidates = data.make_eval_candidates(
            "validation", config.recommendation_list_length, config.seed
        )
        self.negative_pool: Tensor | None = None
        self.transport_items: Tensor | None = None
        self.transport_weights: Tensor | None = None
        self.pool_diagnostics: dict[str, float | int | str] = {}
        self.click_centroids: Tensor | None = None
        self.exposure_centroids: Tensor | None = None
        if config.selection_mode == "safe-hard":
            self.click_centroids, self.exposure_centroids = self._train_centroids()

    def _autocast(self):
        if not self.config.amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _training_indices(self) -> np.ndarray:
        size = self.data.train_users.size
        maximum = self.config.maximum_train_examples
        if maximum is not None and maximum < size:
            return self.rng.choice(size, size=maximum, replace=False)
        return self.rng.permutation(size)

    def _train_centroids(self) -> tuple[Tensor, Tensor]:
        """Train-only normalized user centroids used as a false-negative proxy."""

        if self.flow is None:
            raise RuntimeError("safe-hard centroids require an exposure flow")
        catalog = self.catalog.detach().cpu().numpy()
        fallback = catalog.mean(axis=0)
        click = np.empty((self.data.num_users, self.ranker.embedding_dim), np.float32)
        exposure = np.empty_like(click)
        for user in range(self.data.num_users):
            clicked_items = self.data.train_clicks.for_user(user)
            exposed_items = self.data.train_exposures.for_user(user)
            click[user] = (
                catalog[clicked_items].mean(axis=0)
                if clicked_items.size
                else fallback
            )
            exposure[user] = (
                catalog[exposed_items].mean(axis=0)
                if exposed_items.size
                else fallback
            )
        mean = self.flow.catalog_mean.detach().cpu()
        scale = self.flow.catalog_scale.detach().cpu()
        return (
            (torch.from_numpy(click) - mean) / scale,
            (torch.from_numpy(exposure) - mean) / scale,
        )

    @torch.inference_mode()
    def refresh_negative_pool(self, epoch: int) -> dict[str, float | int | str]:
        """Generate a bounded reusable pool without mapping to catalog IDs."""

        if self.config.negative_source not in {"flow", "soft-flow", "flow-dns"}:
            self.negative_pool = None
            self.pool_diagnostics = {"pool_source": "uniform_catalog"}
            return self.pool_diagnostics
        assert self.flow is not None
        self.flow.eval()
        candidates = self.config.candidates_per_user
        flat_users = torch.arange(self.data.num_users, dtype=torch.int64).repeat_interleave(
            candidates
        )
        flat_pool = torch.empty(
            flat_users.numel(), self.ranker.embedding_dim, dtype=torch.float32
        )
        generator = torch.Generator(device=self.device).manual_seed(
            self.config.seed + 10_000 + epoch
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        for begin in range(0, flat_users.numel(), self.config.pool_batch_size):
            end = min(begin + self.config.pool_batch_size, flat_users.numel())
            users = flat_users[begin:end].to(self.device, non_blocking=True)
            noise = torch.randn(
                users.numel(),
                self.ranker.embedding_dim,
                generator=generator,
                device=self.device,
            )
            with self._autocast():
                generated = self.flow.sample(
                    users,
                    steps=self.config.flow_steps,
                    solver=self.config.flow_solver,
                    noise=noise,
                )
            flat_pool[begin:end].copy_(generated.float().cpu())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - start
        self.negative_pool = flat_pool.reshape(
            self.data.num_users, candidates, self.ranker.embedding_dim
        )
        diagnostics = self._pool_manifold_diagnostics(epoch)
        if self.config.negative_source in {"soft-flow", "flow-dns"}:
            diagnostics.update(self._build_soft_transport())
        diagnostics.update(
            {
                "pool_source": (
                    "soft_catalog_transport"
                    if self.config.negative_source == "soft-flow"
                    else (
                        "flow_dns"
                        if self.config.negative_source == "flow-dns"
                        else "continuous_flow"
                    )
                ),
                "pool_embeddings": int(flat_users.numel()),
                "pool_seconds": seconds,
                "pool_embeddings_per_second": flat_users.numel()
                / max(seconds, 1e-12),
            }
        )
        self.pool_diagnostics = diagnostics
        return diagnostics

    def _build_soft_transport(self) -> dict[str, float | int]:
        """Map each endpoint to a soft neighborhood of train-exposed item IDs.

        The candidate set is restricted to the same user's exposed non-clicks,
        so no validation/test behavior can enter the mapping.  Per-endpoint
        inverse temperatures are increased only until a fixed ESS floor is met.
        """

        assert self.negative_pool is not None and self.flow is not None
        users = self.data.num_users
        candidates = self.config.candidates_per_user
        neighbors = self.config.transport_neighbors
        mean = self.flow.catalog_mean.detach().cpu()
        scale = self.flow.catalog_scale.detach().cpu()
        catalog = ((self.catalog.detach().cpu() - mean) / scale).numpy()
        pool = ((self.negative_pool - mean) / scale).numpy()
        item_output = np.empty((users, candidates, neighbors), dtype=np.int32)
        weight_output = np.zeros((users, candidates, neighbors), dtype=np.float32)
        ess_total = 0.0
        distance_total = 0.0
        maximum_weight_total = 0.0
        mapped = 0
        exposure_backed = 0
        start = time.perf_counter()
        for user in range(users):
            exposed = self.data.train_exposures.for_user(user)
            if exposed.size:
                choices = exposed.astype(np.int32, copy=False)
                exposure_backed += candidates
            else:
                choices = self.sampler.uniform(
                    np.asarray([user], dtype=np.int32), neighbors
                ).reshape(-1)
            available = min(neighbors, choices.size)
            distances = np.square(
                pool[user, :, None, :] - catalog[choices][None, :, :]
            ).mean(axis=2)
            if available < choices.size:
                positions = np.argpartition(
                    distances, kth=available - 1, axis=1
                )[:, :available]
            else:
                positions = np.broadcast_to(
                    np.arange(choices.size, dtype=np.int64),
                    (candidates, choices.size),
                ).copy()
            selected_distances = np.take_along_axis(distances, positions, axis=1)
            ordering = np.argsort(selected_distances, axis=1)
            positions = np.take_along_axis(positions, ordering, axis=1)
            selected_distances = np.take_along_axis(
                selected_distances, ordering, axis=1
            )
            selected_items = choices[positions]
            item_output[user] = selected_items[:, :1]
            item_output[user, :, :available] = selected_items

            shifted = selected_distances - selected_distances[:, :1]
            low = np.zeros((candidates, 1), dtype=np.float32)
            high = np.full(
                (candidates, 1),
                self.config.transport_max_inverse_temperature,
                dtype=np.float32,
            )
            target_ess = self.config.transport_minimum_ess_ratio * available
            for _ in range(16):
                middle = 0.5 * (low + high)
                logits = -middle * shifted
                logits -= logits.max(axis=1, keepdims=True)
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                ess = 1.0 / np.square(probabilities).sum(axis=1, keepdims=True)
                feasible = ess >= target_ess
                low = np.where(feasible, middle, low)
                high = np.where(feasible, high, middle)
            logits = -low * shifted
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            weight_output[user, :, :available] = probabilities
            ess = 1.0 / np.square(probabilities).sum(axis=1)
            ess_total += float((ess / available).sum())
            distance_total += float(
                (probabilities * np.sqrt(selected_distances)).sum()
            )
            maximum_weight_total += float(probabilities.max(axis=1).sum())
            mapped += candidates

        self.transport_items = torch.from_numpy(item_output)
        self.transport_weights = torch.from_numpy(weight_output)
        return {
            "transport_mapped_endpoints": mapped,
            "transport_neighbors": neighbors,
            "transport_exposure_backed_rate": exposure_backed / max(mapped, 1),
            "transport_ess_ratio": ess_total / max(mapped, 1),
            "transport_weighted_distance_mean": distance_total / max(mapped, 1),
            "transport_maximum_weight_mean": maximum_weight_total / max(mapped, 1),
            "transport_seconds": time.perf_counter() - start,
        }

    @torch.inference_mode()
    def _pool_manifold_diagnostics(self, epoch: int) -> dict[str, float | int]:
        """Measure the rounding gap for diagnosis only; training stays continuous."""

        assert self.negative_pool is not None and self.flow is not None
        flat = self.negative_pool.reshape(-1, self.ranker.embedding_dim)
        count = min(self.config.diagnostics_examples, flat.shape[0])
        selected = np.random.default_rng(self.config.seed + 20_000 + epoch).choice(
            flat.shape[0], size=count, replace=False
        )
        generated = self.flow.normalize(flat[selected].to(self.device)).float()
        catalog = self.flow.normalize(self.catalog).float()
        nearest_distances: list[Tensor] = []
        nearest_items: list[Tensor] = []
        for begin in range(0, count, 128):
            distances = torch.cdist(generated[begin : begin + 128], catalog)
            values, indices = distances.min(dim=1)
            nearest_distances.append(values.cpu())
            nearest_items.append(indices.cpu())
        nearest = torch.cat(nearest_items).numpy().astype(np.int32, copy=False)
        users = (selected // self.config.candidates_per_user).astype(
            np.int32, copy=False
        )
        is_click = self.data.train_clicks.contains(users, nearest)
        is_exposure = self.data.train_exposures.contains(users, nearest)
        distance = torch.cat(nearest_distances).numpy()
        return {
            "diagnostic_examples": count,
            "nearest_catalog_distance_mean": float(distance.mean()),
            "nearest_catalog_distance_p95": float(np.quantile(distance, 0.95)),
            "nearest_train_click_rate": float(is_click.mean()),
            "nearest_train_exposure_rate": float(is_exposure.mean()),
            "nearest_unique_item_ratio": float(np.unique(nearest).size / count),
        }

    def _negative_embeddings(
        self, users_np: np.ndarray, users: Tensor
    ) -> tuple[Tensor, Tensor | None, dict[str, float]]:
        if self.config.negative_source == "uniform":
            items_np = self.sampler.uniform(users_np, 1).reshape(-1)
            items = torch.from_numpy(items_np.astype(np.int64, copy=False)).to(
                self.device, non_blocking=True
            )
            embeddings = (
                self.catalog[items]
                if self.config.freeze_item_embeddings
                else self.ranker.item_embedding(items)
            )
            return embeddings, None, {}
        if self.negative_pool is None:
            raise RuntimeError("continuous negative pool has not been generated")
        user_indices = torch.from_numpy(users_np.astype(np.int64, copy=False))
        if self.config.negative_source == "flow-dns":
            return self._flow_dns_negative_embeddings(users_np, users, user_indices)
        candidates = self.negative_pool[user_indices].to(
            self.device, non_blocking=True
        )
        if self.config.selection_mode == "random":
            choices = self.rng.integers(
                self.config.candidates_per_user, size=users_np.size
            )
            candidate_indices = torch.from_numpy(
                choices.astype(np.int64, copy=False)
            )
            device_candidate_indices = candidate_indices.to(self.device)
            embeddings = candidates[
                torch.arange(users.numel(), device=self.device),
                device_candidate_indices,
            ]
            if self.config.negative_source == "soft-flow":
                assert self.transport_items is not None
                assert self.transport_weights is not None
                items = self.transport_items[user_indices, candidate_indices].to(
                    self.device, non_blocking=True
                ).long()
                weights = self.transport_weights[
                    user_indices, candidate_indices
                ].to(self.device, non_blocking=True)
                embeddings = (
                    self.catalog[items]
                    if self.config.freeze_item_embeddings
                    else self.ranker.item_embedding(items)
                )
                return embeddings, weights, {}
            return embeddings, None, {}

        assert self.flow is not None
        assert self.click_centroids is not None
        assert self.exposure_centroids is not None
        with torch.no_grad():
            scores = self.ranker.score_embedding_candidates(users, candidates).float()
            centered = scores - scores.mean(dim=1, keepdim=True)
            score_scale = centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(
                1e-6
            )
            hardness = torch.relu(centered / score_scale)
            normalized = self.flow.normalize(candidates).float()
            click_center = self.click_centroids[user_indices].to(self.device)
            exposure_center = self.exposure_centroids[user_indices].to(self.device)
            click_distance = (
                normalized - click_center[:, None, :]
            ).square().mean(dim=2)
            exposure_distance = (
                normalized - exposure_center[:, None, :]
            ).square().mean(dim=2)
            safety = torch.sigmoid(
                (click_distance - exposure_distance)
                / self.config.safety_temperature
            )
            utility = hardness * safety

            target_ess = (
                self.config.minimum_ess_ratio * self.config.candidates_per_user
            )
            low = utility.new_zeros(users.numel(), 1)
            high = utility.new_full((users.numel(), 1), self.config.hardness_beta)
            for _ in range(16):
                middle = 0.5 * (low + high)
                probabilities = torch.softmax(middle * utility, dim=1)
                ess = probabilities.square().sum(dim=1, keepdim=True).reciprocal()
                feasible = ess >= target_ess
                low = torch.where(feasible, middle, low)
                high = torch.where(feasible, high, middle)
            probabilities = torch.softmax(low * utility, dim=1)
            ess = probabilities.square().sum(dim=1).reciprocal()
            kl = (
                probabilities
                * torch.log(
                    probabilities.mul(self.config.candidates_per_user).clamp_min(1e-12)
                )
            ).sum(dim=1)
            candidate_indices = torch.multinomial(probabilities, 1).squeeze(1)
            rows = torch.arange(users.numel(), device=self.device)
            embeddings = candidates[rows, candidate_indices]
            diagnostics = {
                "selection_ess_ratio": float(
                    (ess / self.config.candidates_per_user).mean()
                ),
                "selection_kl_to_flow_pool": float(kl.mean()),
                "selected_standardized_hardness": float(
                    hardness[rows, candidate_indices].mean()
                ),
                "selected_safety_support": float(
                    safety[rows, candidate_indices].mean()
                ),
                "effective_hardness_beta": float(low.mean()),
            }
        if self.config.negative_source == "soft-flow":
            assert self.transport_items is not None
            assert self.transport_weights is not None
            cpu_candidate_indices = candidate_indices.detach().cpu()
            items = self.transport_items[user_indices, cpu_candidate_indices].to(
                self.device, non_blocking=True
            ).long()
            weights = self.transport_weights[
                user_indices, cpu_candidate_indices
            ].to(self.device, non_blocking=True)
            embeddings = (
                self.catalog[items]
                if self.config.freeze_item_embeddings
                else self.ranker.item_embedding(items)
            )
            return embeddings, weights, diagnostics
        return embeddings, None, diagnostics

    def _flow_dns_negative_embeddings(
        self,
        users_np: np.ndarray,
        users: Tensor,
        user_indices: Tensor,
    ) -> tuple[Tensor, None, dict[str, float]]:
        """Select the hardest item from 29 uniform and 1 Flow proposal by default."""

        assert self.transport_items is not None
        assert self.transport_weights is not None
        batch = users_np.size
        flow_count = self.config.dns_flow_candidates
        endpoint_choices_np = self.rng.integers(
            self.config.candidates_per_user, size=(batch, flow_count)
        )
        endpoint_choices = torch.from_numpy(
            endpoint_choices_np.astype(np.int64, copy=False)
        )
        expanded_users = user_indices[:, None].expand(-1, flow_count)
        mapped_items = self.transport_items[expanded_users, endpoint_choices]
        mapped_weights = self.transport_weights[expanded_users, endpoint_choices]
        neighbor_choices = torch.multinomial(
            mapped_weights.reshape(-1, self.config.transport_neighbors), 1
        ).reshape(batch, flow_count)
        flow_items = mapped_items.gather(2, neighbor_choices[:, :, None]).squeeze(2)
        uniform_np = self.sampler.uniform(
            users_np,
            self.config.dns_uniform_candidates,
            exclude_exposures=True,
        )
        uniform_items = torch.from_numpy(
            uniform_np.astype(np.int64, copy=False)
        )
        candidate_items_cpu = torch.cat((uniform_items, flow_items.long()), dim=1)
        candidate_items = candidate_items_cpu.to(self.device, non_blocking=True)
        with torch.no_grad():
            candidate_scores = self.ranker.score_candidates(users, candidate_items)
            selected_positions = candidate_scores.argmax(dim=1)
        rows = torch.arange(batch, device=self.device)
        selected_items = candidate_items[rows, selected_positions]
        selected_np = selected_items.cpu().numpy().astype(np.int32, copy=False)
        selected_from_flow = selected_positions >= self.config.dns_uniform_candidates
        diagnostics = {
            "dns_selected_flow_rate": float(selected_from_flow.float().mean()),
            "dns_selected_exposure_rate": float(
                self.data.train_exposures.contains(users_np, selected_np).mean()
            ),
            "dns_flow_candidate_ratio": flow_count
            / (flow_count + self.config.dns_uniform_candidates),
        }
        embeddings = (
            self.catalog[selected_items]
            if self.config.freeze_item_embeddings
            else self.ranker.item_embedding(selected_items)
        )
        return embeddings, None, diagnostics

    def train_epoch(self, epoch: int) -> dict[str, float | int]:
        if self.config.negative_source in {"flow", "soft-flow", "flow-dns"} and (
            self.negative_pool is None
            or (epoch - 1) % self.config.pool_refresh_epochs == 0
        ):
            self.refresh_negative_pool(epoch)
        self.ranker.train()
        order = self._training_indices()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        loss_total = 0.0
        positive_total = 0.0
        negative_total = 0.0
        selection_totals: dict[str, float] = {}
        processed = 0
        for begin in range(0, order.size, self.config.batch_size):
            selected = order[begin : min(begin + self.config.batch_size, order.size)]
            users_np = self.data.train_users[selected]
            positives_np = self.data.train_items[selected]
            users = torch.from_numpy(users_np.astype(np.int64, copy=False)).to(
                self.device, non_blocking=True
            )
            positives = torch.from_numpy(
                positives_np.astype(np.int64, copy=False)
            ).to(self.device, non_blocking=True)
            negative_embeddings, negative_weights, selection = self._negative_embeddings(
                users_np, users
            )
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                positive_scores = self.ranker.score(users, positives)
                if negative_weights is None:
                    negative_scores = self.ranker.score_embeddings(
                        users, negative_embeddings
                    )
                    difference = positive_scores - negative_scores
                    loss = (
                        F.softplus(-difference).sum()
                        if self.config.ranking_loss == "bpr"
                        else hard_bpr_loss(
                            difference,
                            a=self.config.hard_bpr_a,
                            b=self.config.hard_bpr_b,
                            c=self.config.hard_bpr_c,
                        ).sum()
                    )
                    negative_score_sum = negative_scores.detach().float().sum()
                else:
                    negative_scores = self.ranker.score_embedding_candidates(
                        users, negative_embeddings
                    )
                    difference = positive_scores[:, None] - negative_scores
                    pair_losses = (
                        F.softplus(-difference)
                        if self.config.ranking_loss == "bpr"
                        else hard_bpr_loss(
                            difference,
                            a=self.config.hard_bpr_a,
                            b=self.config.hard_bpr_b,
                            c=self.config.hard_bpr_c,
                        )
                    )
                    loss = (negative_weights * pair_losses).sum()
                    negative_score_sum = (
                        negative_weights * negative_scores.detach().float()
                    ).sum()
                if self.config.discriminator_reg:
                    regularization = self.ranker.user_embedding(users).square().sum()
                    if not self.config.freeze_item_embeddings:
                        regularization = regularization + self.ranker.item_embedding(
                            positives
                        ).square().sum()
                        if negative_weights is None:
                            regularization = (
                                regularization + negative_embeddings.square().sum()
                            )
                        else:
                            regularization = regularization + (
                                negative_weights[:, :, None]
                                * negative_embeddings.square()
                            ).sum()
                    loss = loss + 0.5 * self.config.discriminator_reg * regularization
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.ranker.parameters() if p.requires_grad),
                    self.config.gradient_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            count = int(users.numel())
            loss_total += float(loss.detach())
            positive_total += float(positive_scores.detach().float().sum())
            negative_total += float(negative_score_sum)
            for key, value in selection.items():
                selection_totals[key] = selection_totals.get(key, 0.0) + value * count
            processed += count
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - start
        result: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": loss_total / max(processed, 1),
            "positive_score_mean": positive_total / max(processed, 1),
            "negative_score_mean": negative_total / max(processed, 1),
            "training_examples": processed,
            "training_seconds": seconds,
            "training_examples_per_second": processed / max(seconds, 1e-12),
        }
        result.update(
            {
                key: value / max(processed, 1)
                for key, value in selection_totals.items()
            }
        )
        return result

    def evaluate_validation(self) -> EvalResult:
        return evaluate_list(
            self.ranker,
            self.eval_candidates,
            device=self.device,
            batch_size=self.config.eval_batch_size,
        )

    def _checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        validation: EvalResult,
        history: list[dict[str, Any]],
    ) -> None:
        torch.save(
            {
                "format_version": 1,
                "method": (
                    f"flow-dns-{self.config.ranking_loss}"
                    if self.config.negative_source == "flow-dns"
                    else (
                        "soft-transport-flow-bpr"
                        if self.config.negative_source == "soft-flow"
                        else "continuous-flow-bpr"
                    )
                ),
                "config": self.config.to_dict(),
                "epoch": epoch,
                "validation": validation.to_dict(),
                "history": history,
                "item_embeddings_frozen": self.config.freeze_item_embeddings,
                "ranker": self.ranker.state_dict(),
            },
            path,
        )

    def fit(self, output_dir: str | Path) -> list[dict[str, Any]]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        initial = self.evaluate_validation()
        history: list[dict[str, Any]] = [
            {
                "epoch": 0,
                **{
                    f"validation_{key}": value
                    for key, value in initial.to_dict().items()
                },
            }
        ]
        print(f"Initial validation: {format_result(initial)}", flush=True)
        best_ndcg = initial.ndcg
        best_epoch = 0
        stale = 0
        self._checkpoint(
            output / "best.pt", epoch=0, validation=initial, history=history
        )

        for epoch in range(1, self.config.epochs + 1):
            training = self.train_epoch(epoch)
            validation = self.evaluate_validation()
            record: dict[str, Any] = {
                **training,
                **{
                    f"validation_{key}": value
                    for key, value in validation.to_dict().items()
                },
            }
            if self.config.negative_source in {"flow", "soft-flow", "flow-dns"} and (
                (epoch - 1) % self.config.pool_refresh_epochs == 0
            ):
                record.update(self.pool_diagnostics)
            history.append(record)
            (output / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"Continuous ranker {epoch}: loss={training['training_loss']:.6f} "
                f"{format_result(validation)}",
                flush=True,
            )
            if validation.ndcg > best_ndcg + self.config.minimum_ndcg_improvement:
                best_ndcg = validation.ndcg
                best_epoch = epoch
                stale = 0
                self._checkpoint(
                    output / "best.pt",
                    epoch=epoch,
                    validation=validation,
                    history=history,
                )
            else:
                stale += 1
            self._checkpoint(
                output / "last.pt",
                epoch=epoch,
                validation=validation,
                history=history,
            )
            if (
                epoch >= self.config.minimum_epochs
                and stale >= self.config.early_stopping_patience
            ):
                break

        summary = {
            "best_epoch": best_epoch,
            "best_validation_ndcg": best_ndcg,
            "epochs_completed": len(history) - 1,
            "official_test_used": False,
            "item_embeddings_frozen": self.config.freeze_item_embeddings,
            "continuous_training_without_catalog_mapping": (
                self.config.negative_source == "flow"
            ),
            "soft_catalog_transport": self.config.negative_source
            in {"soft-flow", "flow-dns"},
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return history
