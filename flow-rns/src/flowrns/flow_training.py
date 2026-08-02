from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from reinforcens.data import InteractionData
from reinforcens.trainer import resolve_device, seed_everything

from .flow import ConditionalExposureFlow, ExposureFlowConfig


@dataclass(slots=True)
class FlowTrainingConfig:
    batch_size: int = 2_048
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    minimum_epochs: int = 10
    early_stopping_patience: int = 5
    minimum_relative_improvement: float = 0.005
    validation_fraction: float = 0.02
    validation_examples: int = 131_072
    diagnostics_examples: int = 4_096
    diagnostics_projections: int = 64
    maximum_training_exposures: int | None = None
    seed: int = 1
    device: str = "auto"
    amp: bool = False
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if min(
            self.batch_size,
            self.epochs,
            self.minimum_epochs,
            self.early_stopping_patience,
            self.validation_examples,
            self.diagnostics_examples,
            self.diagnostics_projections,
        ) <= 0:
            raise ValueError("flow training counts must be positive")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs cannot exceed epochs")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid flow optimizer configuration")
        if not 0 <= self.minimum_relative_improvement < 1:
            raise ValueError("minimum_relative_improvement must be in [0, 1)")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if (
            self.maximum_training_exposures is not None
            and self.maximum_training_exposures <= 0
        ):
            raise ValueError("maximum_training_exposures must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_train_exposures(
    exposure_keys: np.ndarray,
    *,
    validation_fraction: float,
    validation_examples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic holdout using train-split exposure pairs only."""

    keys = np.asarray(exposure_keys, dtype=np.int64)
    if keys.size < 2:
        raise ValueError("at least two train exposure pairs are required")
    requested = max(1, int(keys.size * validation_fraction))
    count = min(requested, validation_examples, keys.size - 1)
    order = np.random.default_rng(seed).permutation(keys.size)
    validation = keys[order[:count]].copy()
    training = keys[order[count:]].copy()
    return training, validation


def sliced_wasserstein_distance(
    first: Tensor,
    second: Tensor,
    *,
    projections: int,
    generator: torch.Generator,
) -> Tensor:
    """Mean one-dimensional Wasserstein-2 distance over random projections."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("sliced Wasserstein inputs must have equal matrix shapes")
    directions = torch.randn(
        projections,
        first.shape[1],
        generator=generator,
        device=first.device,
        dtype=first.dtype,
    )
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    first_projection = torch.sort(first @ directions.T, dim=0).values
    second_projection = torch.sort(second @ directions.T, dim=0).values
    return (first_projection - second_projection).square().mean().sqrt()


class ExposureFlowTrainer:
    """Full-data trainer with train-only holdout and distribution diagnostics."""

    def __init__(
        self,
        data: InteractionData,
        catalog_embeddings: Tensor,
        flow_config: ExposureFlowConfig,
        training_config: FlowTrainingConfig,
    ) -> None:
        if catalog_embeddings.shape != (data.num_items, flow_config.embedding_dim):
            raise ValueError("catalog embedding shape does not match data/flow config")
        if flow_config.num_users != data.num_users:
            raise ValueError("flow user count does not match data")
        self.data = data
        self.flow_config = flow_config
        self.training_config = training_config
        self.device = resolve_device(training_config.device)
        if training_config.amp and self.device.type != "cuda":
            raise ValueError("flow AMP requires CUDA")
        seed_everything(training_config.seed)
        self.rng = np.random.default_rng(training_config.seed + 101)
        self.catalog_embeddings = catalog_embeddings.detach().float().to(self.device)
        self.flow = ConditionalExposureFlow(
            flow_config, self.catalog_embeddings
        ).to(self.device)
        self.initialization: dict[str, Any] | None = None
        self.optimizer = torch.optim.AdamW(
            self.flow.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=training_config.amp
        )
        self.training_keys, self.validation_keys = split_train_exposures(
            data.train_exposures.keys,
            validation_fraction=training_config.validation_fraction,
            validation_examples=training_config.validation_examples,
            seed=training_config.seed + 103,
        )
        maximum = training_config.maximum_training_exposures
        if maximum is not None and self.training_keys.size > maximum:
            selected = np.random.default_rng(
                training_config.seed + 107
            ).choice(self.training_keys.size, size=maximum, replace=False)
            self.training_keys = self.training_keys[selected].copy()

    def initialize_from_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Warm-start model weights while intentionally creating a new optimizer."""

        source = Path(path).expanduser().resolve()
        payload = torch.load(source, map_location=self.device, weights_only=False)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported exposure-flow checkpoint format")
        source_config = ExposureFlowConfig.from_dict(payload["flow_config"])
        if source_config != self.flow_config:
            raise ValueError("warm-start flow architecture does not match")
        self.flow.load_state_dict(payload["flow"])
        self.initialization = {
            "checkpoint": str(source),
            "source_epoch": payload.get("epoch"),
            "source_metrics": payload.get("metrics"),
            "optimizer_state_reused": False,
        }
        return self.initialization

    def _autocast(self):
        if not self.training_config.amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _keys_to_tensors(self, keys: np.ndarray) -> tuple[Tensor, Tensor]:
        users = torch.from_numpy(
            (keys // self.data.num_items).astype(np.int64, copy=False)
        ).to(self.device, non_blocking=True)
        items = torch.from_numpy(
            (keys % self.data.num_items).astype(np.int64, copy=False)
        ).to(self.device, non_blocking=True)
        return users, items

    def _training_epoch(self) -> dict[str, float | int]:
        self.flow.train()
        order = self.rng.permutation(self.training_keys.size)
        loss_total = 0.0
        velocity_total = 0.0
        endpoint_total = 0.0
        processed = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        for begin in range(0, order.size, self.training_config.batch_size):
            selected = self.training_keys[
                order[begin : min(begin + self.training_config.batch_size, order.size)]
            ]
            users, items = self._keys_to_tensors(selected)
            targets = self.catalog_embeddings[items]
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss, metrics = self.flow.flow_matching_loss(users, targets)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.flow.parameters(), self.training_config.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            count = int(users.numel())
            loss_total += float(loss.detach()) * count
            velocity_total += float(metrics["velocity_mse"]) * count
            endpoint_total += float(metrics["endpoint_mse"]) * count
            processed += count
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - start
        return {
            "training_loss": loss_total / processed,
            "training_velocity_mse": velocity_total / processed,
            "training_endpoint_mse": endpoint_total / processed,
            "training_examples": processed,
            "training_seconds": seconds,
            "training_examples_per_second": processed / seconds,
            "peak_memory_mib": (
                torch.cuda.max_memory_allocated(self.device) / 2**20
                if self.device.type == "cuda"
                else 0.0
            ),
        }

    @torch.inference_mode()
    def _validation_epoch(self, epoch: int) -> dict[str, float | int]:
        self.flow.eval()
        loss_total = 0.0
        velocity_total = 0.0
        endpoint_total = 0.0
        processed = 0
        generator = torch.Generator(device=self.device).manual_seed(
            self.training_config.seed + 10_000
        )
        for begin in range(0, self.validation_keys.size, self.training_config.batch_size):
            selected = self.validation_keys[
                begin : min(begin + self.training_config.batch_size, self.validation_keys.size)
            ]
            users, items = self._keys_to_tensors(selected)
            targets = self.catalog_embeddings[items]
            times = torch.rand(users.numel(), generator=generator, device=self.device)
            noise = torch.randn(
                users.numel(),
                self.flow_config.embedding_dim,
                generator=generator,
                device=self.device,
            )
            with self._autocast():
                loss, metrics = self.flow.flow_matching_loss(
                    users, targets, times=times, noise=noise
                )
            count = int(users.numel())
            loss_total += float(loss) * count
            velocity_total += float(metrics["velocity_mse"]) * count
            endpoint_total += float(metrics["endpoint_mse"]) * count
            processed += count
        result: dict[str, float | int] = {
            "epoch": epoch,
            "validation_loss": loss_total / processed,
            "validation_velocity_mse": velocity_total / processed,
            "validation_endpoint_mse": endpoint_total / processed,
            "validation_examples": processed,
        }
        result.update(self._distribution_diagnostics())
        return result

    @torch.inference_mode()
    def _distribution_diagnostics(self) -> dict[str, float]:
        count = min(
            self.training_config.diagnostics_examples,
            self.validation_keys.size,
        )
        selected = self.validation_keys[:count]
        users, items = self._keys_to_tensors(selected)
        target = self.flow.normalize(self.catalog_embeddings[items]).float()
        generator = torch.Generator(device=self.device).manual_seed(
            self.training_config.seed + 20_000
        )
        noise = torch.randn(
            count,
            self.flow_config.embedding_dim,
            generator=generator,
            device=self.device,
        )
        generated = self.flow.integrate(
            users,
            noise,
            steps=self.flow_config.sampling_steps,
            solver="heun",
        ).float()
        projection_generator = torch.Generator(device=self.device).manual_seed(
            self.training_config.seed + 30_000
        )
        swd = sliced_wasserstein_distance(
            generated,
            target,
            projections=self.training_config.diagnostics_projections,
            generator=projection_generator,
        )
        mean_rmse = (generated.mean(0) - target.mean(0)).square().mean().sqrt()
        std_rmse = (
            generated.std(0, unbiased=False) - target.std(0, unbiased=False)
        ).square().mean().sqrt()
        return {
            "diagnostic_sliced_wasserstein": float(swd),
            "diagnostic_mean_rmse": float(mean_rmse),
            "diagnostic_std_rmse": float(std_rmse),
            "diagnostic_generated_norm": float(generated.norm(dim=1).mean()),
            "diagnostic_target_norm": float(target.norm(dim=1).mean()),
        }

    def _checkpoint_payload(
        self, epoch: int, metrics: dict[str, Any], history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "format_version": 1,
            "flow_config": self.flow_config.to_dict(),
            "training_config": self.training_config.to_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "history": history,
            "training_exposures": int(self.training_keys.size),
            "validation_exposures": int(self.validation_keys.size),
            "initialization": self.initialization,
            "flow": self.flow.state_dict(),
        }

    def fit(self, output_dir: str | Path) -> list[dict[str, Any]]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(
            json.dumps(
                {
                    "flow": self.flow_config.to_dict(),
                    "training": self.training_config.to_dict(),
                    "num_parameters": self.flow.num_parameters,
                    "data_source": self.data.source,
                    "train_split_only": True,
                    "initialization": self.initialization,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        history: list[dict[str, Any]] = []
        best_loss = math.inf
        material_reference_loss = math.inf
        stale = 0
        for epoch in range(1, self.training_config.epochs + 1):
            training = self._training_epoch()
            validation = self._validation_epoch(epoch)
            record = {**validation, **training}
            history.append(record)
            (output / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            loss = float(validation["validation_loss"])
            relative = (best_loss - loss) / max(abs(best_loss), 1e-12)
            improved = not math.isfinite(best_loss) or relative > 0
            material_relative = (
                (material_reference_loss - loss)
                / max(abs(material_reference_loss), 1e-12)
            )
            materially_improved = not math.isfinite(
                material_reference_loss
            ) or material_relative >= self.training_config.minimum_relative_improvement
            if improved:
                best_loss = loss
                torch.save(
                    self._checkpoint_payload(epoch, validation, history),
                    output / "best.pt",
                )
            if materially_improved:
                material_reference_loss = loss
                stale = 0
            elif epoch >= self.training_config.minimum_epochs:
                stale += 1
            torch.save(
                self._checkpoint_payload(epoch, validation, history),
                output / "last.pt",
            )
            print(
                f"Flow epoch {epoch}: train={training['training_loss']:.6f} "
                f"val={loss:.6f} swd={validation['diagnostic_sliced_wasserstein']:.4f} "
                f"{training['training_examples_per_second']:,.0f} exposures/s "
                f"memory={training['peak_memory_mib']:.0f} MiB",
                flush=True,
            )
            if (
                epoch >= self.training_config.minimum_epochs
                and stale >= self.training_config.early_stopping_patience
            ):
                print(
                    "Flow early stopping: holdout loss has not materially improved "
                    f"for {stale} epochs",
                    flush=True,
                )
                break
        return history


def load_exposure_flow(
    path: str | Path,
    catalog_embeddings: Tensor,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ConditionalExposureFlow, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported exposure-flow checkpoint format")
    config = ExposureFlowConfig.from_dict(payload["flow_config"])
    flow = ConditionalExposureFlow(config, catalog_embeddings).to(device)
    flow.load_state_dict(payload["flow"])
    metadata = {
        key: payload.get(key)
        for key in (
            "epoch",
            "metrics",
            "history",
            "training_config",
            "initialization",
            "training_exposures",
            "validation_exposures",
        )
    }
    return flow, metadata
