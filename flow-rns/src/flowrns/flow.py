from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(slots=True)
class ExposureFlowConfig:
    """Architecture and solver options for a user-conditioned exposure flow."""

    num_users: int = 16_015
    embedding_dim: int = 32
    user_dim: int = 128
    time_dim: int = 128
    hidden_dim: int = 512
    depth: int = 8
    sampling_steps: int = 16
    reconstruction_weight: float = 0.25

    def __post_init__(self) -> None:
        if min(
            self.num_users,
            self.embedding_dim,
            self.user_dim,
            self.hidden_dim,
            self.depth,
            self.sampling_steps,
        ) <= 0:
            raise ValueError("flow dimensions, depth, and sampling steps must be positive")
        if self.time_dim <= 0 or self.time_dim % 2:
            raise ValueError("time_dim must be a positive even integer")
        if self.reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExposureFlowConfig":
        return cls(**values)


def sinusoidal_time_embedding(times: Tensor, dimension: int) -> Tensor:
    """Fourier features for times in ``[0, 1]``."""

    if dimension <= 0 or dimension % 2:
        raise ValueError("time embedding dimension must be positive and even")
    times = times.reshape(-1).float()
    half = dimension // 2
    scale = -math.log(10_000.0) / max(half - 1, 1)
    frequencies = torch.exp(
        torch.arange(half, device=times.device, dtype=times.dtype) * scale
    )
    angles = 2.0 * math.pi * times[:, None] * frequencies[None, :]
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


class FiLMResidualBlock(nn.Module):
    """Pre-normalized gated residual block with repeated user/time conditioning."""

    def __init__(self, hidden_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_dim)
        self.condition = nn.Linear(condition_dim, 2 * hidden_dim)
        self.input_projection = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        normalized = self.normalization(values)
        normalized = normalized * (1.0 + 0.1 * torch.tanh(scale)) + shift
        value, gate = self.input_projection(normalized).chunk(2, dim=-1)
        update = self.output_projection(value * F.silu(gate))
        return (values + update) / math.sqrt(2.0)


class ConditionalExposureFlow(nn.Module):
    """Conditional flow matching model over a fixed catalog embedding geometry.

    The target distribution contains only train-split exposed, non-clicked items.
    Item standardization is part of the model state so training and sampling use
    exactly the same coordinate transform.
    """

    def __init__(self, config: ExposureFlowConfig, catalog_embeddings: Tensor) -> None:
        super().__init__()
        if catalog_embeddings.ndim != 2 or catalog_embeddings.shape[1] != config.embedding_dim:
            raise ValueError(
                "catalog_embeddings must be [num_items, embedding_dim]"
            )
        self.config = config
        catalog = catalog_embeddings.detach().float()
        mean = catalog.mean(dim=0)
        scale = catalog.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.register_buffer("catalog_mean", mean)
        self.register_buffer("catalog_scale", scale)

        self.user_embedding = nn.Embedding(config.num_users, config.user_dim)
        condition_dim = config.user_dim + config.time_dim
        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(config.hidden_dim, condition_dim)
            for _ in range(config.depth)
        )
        self.output_normalization = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def normalize(self, embeddings: Tensor) -> Tensor:
        return (embeddings.float() - self.catalog_mean) / self.catalog_scale

    def denormalize(self, embeddings: Tensor) -> Tensor:
        return embeddings.float() * self.catalog_scale + self.catalog_mean

    def forward(self, users: Tensor, values: Tensor, times: Tensor) -> Tensor:
        if users.ndim != 1 or values.shape != (
            users.numel(),
            self.config.embedding_dim,
        ):
            raise ValueError("users and flow values have incompatible shapes")
        if times.numel() not in {1, users.numel()}:
            raise ValueError("times must contain one value or one value per user")
        if times.numel() == 1:
            times = times.expand(users.numel())
        time = sinusoidal_time_embedding(times, self.config.time_dim)
        condition = torch.cat((self.user_embedding(users), time), dim=-1)
        hidden = self.input_projection(values.float())
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output_projection(F.silu(self.output_normalization(hidden)))

    def flow_matching_loss(
        self,
        users: Tensor,
        target_embeddings: Tensor,
        *,
        times: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Rectified conditional-flow objective with endpoint reconstruction."""

        target = self.normalize(target_embeddings)
        if target.shape != (users.numel(), self.config.embedding_dim):
            raise ValueError("target embeddings have incompatible shape")
        if times is None:
            times = torch.rand(users.numel(), device=target.device)
        else:
            times = times.reshape(-1).to(target.device)
        if noise is None:
            noise = torch.randn_like(target)
        else:
            noise = noise.to(target.device).float()
        interpolation = (1.0 - times[:, None]) * noise + times[:, None] * target
        target_velocity = target - noise
        predicted_velocity = self(users, interpolation, times)
        velocity_mse = F.mse_loss(predicted_velocity, target_velocity)
        predicted_endpoint = interpolation + (1.0 - times[:, None]) * predicted_velocity
        endpoint_mse = F.mse_loss(predicted_endpoint, target)
        loss = velocity_mse + self.config.reconstruction_weight * endpoint_mse
        metrics = {
            "velocity_mse": velocity_mse.detach(),
            "endpoint_mse": endpoint_mse.detach(),
        }
        return loss, metrics

    def integrate(
        self,
        users: Tensor,
        noise: Tensor,
        *,
        steps: int | None = None,
        solver: str = "heun",
    ) -> Tensor:
        """Integrate the learned ODE and return normalized endpoints."""

        step_count = self.config.sampling_steps if steps is None else steps
        if step_count <= 0:
            raise ValueError("integration steps must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun")
        values = noise.float()
        delta = 1.0 / step_count
        for index in range(step_count):
            time = values.new_full((users.numel(),), index / step_count)
            velocity = self(users, values, time)
            proposal = values + delta * velocity
            if solver == "euler":
                values = proposal
                continue
            next_time = values.new_full((users.numel(),), (index + 1) / step_count)
            next_velocity = self(users, proposal, next_time)
            values = values + 0.5 * delta * (velocity + next_velocity)
        return values

    @torch.inference_mode()
    def sample(
        self,
        users: Tensor,
        *,
        steps: int | None = None,
        solver: str = "heun",
        noise: Tensor | None = None,
    ) -> Tensor:
        if noise is None:
            noise = torch.randn(
                users.numel(),
                self.config.embedding_dim,
                device=users.device,
            )
        normalized = self.integrate(users, noise, steps=steps, solver=solver)
        return self.denormalize(normalized)
