from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Recommender(nn.Module, ABC):
    num_users: int
    num_items: int
    embedding_dim: int

    @abstractmethod
    def score(self, users: Tensor, items: Tensor) -> Tensor:
        """Score aligned user/item vectors."""

    @abstractmethod
    def score_candidates(self, users: Tensor, candidates: Tensor) -> Tensor:
        """Score a [batch, candidates] item matrix."""

    @abstractmethod
    def all_scores(self, users: Tensor) -> Tensor:
        """Score every item for each user."""

    @abstractmethod
    def feature(self, users: Tensor, items: Tensor) -> Tensor:
        """Return the interaction feature used by the original RNS MMD reward."""

    @abstractmethod
    def selected_l2(
        self,
        users: Tensor,
        *item_groups: Tensor,
        include_weights: bool = True,
    ) -> Tensor:
        """Original-RNS-compatible selected-embedding L2 penalty."""


class BarycentricGenerator(Recommender):
    """Weighted reverse-KL barycenter of a trainable policy and exposure prior."""

    def __init__(self, policy: Recommender, exposure_prior: Recommender) -> None:
        super().__init__()
        policy_shape = (policy.num_users, policy.num_items, policy.embedding_dim)
        prior_shape = (
            exposure_prior.num_users,
            exposure_prior.num_items,
            exposure_prior.embedding_dim,
        )
        if policy_shape != prior_shape:
            raise ValueError("barycentric policy and exposure prior shapes differ")
        self.policy = policy
        self.exposure_prior = exposure_prior
        self.num_users, self.num_items, self.embedding_dim = policy_shape
        self.prior_weight = 0.5

    def set_prior_weight(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError("exposure-prior weight must be in [0, 1]")
        self.prior_weight = float(value)

    def _combine(self, policy: Tensor, exposure: Tensor) -> Tensor:
        return (1.0 - self.prior_weight) * policy + self.prior_weight * exposure

    def score(self, users: Tensor, items: Tensor) -> Tensor:
        return self._combine(
            self.policy.score(users, items),
            self.exposure_prior.score(users, items),
        )

    def score_candidates(self, users: Tensor, candidates: Tensor) -> Tensor:
        return self._combine(
            self.policy.score_candidates(users, candidates),
            self.exposure_prior.score_candidates(users, candidates),
        )

    def all_scores(self, users: Tensor) -> Tensor:
        return self._combine(
            self.policy.all_scores(users), self.exposure_prior.all_scores(users)
        )

    def feature(self, users: Tensor, items: Tensor) -> Tensor:
        return self.policy.feature(users, items)

    def selected_l2(
        self,
        users: Tensor,
        *item_groups: Tensor,
        include_weights: bool = True,
    ) -> Tensor:
        return self.policy.selected_l2(
            users, *item_groups, include_weights=include_weights
        )


def exposure_calibrated_hard_tilt(
    ranker_scores: Tensor, exposure_density_logits: Tensor
) -> Tensor:
    """Parameter-free safe-hard utility for a row of generator candidates.

    NCE with equal positive/negative class priors makes ``sigmoid(density_logit)``
    the exposure posterior.  Positive standardized ranker scores identify hard
    candidates.  Their product therefore boosts hard candidates only in
    proportion to the evidence that they are genuine exposed non-clicks.
    """

    if ranker_scores.shape != exposure_density_logits.shape:
        raise ValueError("ranker scores and density logits must have equal shapes")
    scores = ranker_scores.float()
    centered = scores - scores.mean(dim=1, keepdim=True)
    scale = centered.square().mean(dim=1, keepdim=True).add(1e-12).sqrt()
    standardized_positive_hardness = torch.relu(centered / scale)
    exposure_posterior = torch.sigmoid(exposure_density_logits.float())
    return exposure_posterior * standardized_positive_hardness


class GMF(Recommender):
    """Generalized matrix factorization used by the released RNS model."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.h = nn.Parameter(torch.empty(embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        bound = math.sqrt(3.0 / self.embedding_dim)
        nn.init.uniform_(self.h, -bound, bound)

    def score(self, users: Tensor, items: Tensor) -> Tensor:
        return ((self.user_embedding(users) * self.item_embedding(items)) * self.h).sum(
            dim=-1
        )

    def score_candidates(self, users: Tensor, candidates: Tensor) -> Tensor:
        user = self.user_embedding(users) * self.h
        item = self.item_embedding(candidates)
        return torch.bmm(item, user.unsqueeze(-1)).squeeze(-1)

    def score_embeddings(self, users: Tensor, item_embeddings: Tensor) -> Tensor:
        """Score aligned continuous item embeddings in the GMF catalog space."""

        if item_embeddings.shape != (users.numel(), self.embedding_dim):
            raise ValueError(
                "continuous item embeddings must have shape "
                f"({users.numel()}, {self.embedding_dim})"
            )
        return (
            (self.user_embedding(users) * item_embeddings) * self.h
        ).sum(dim=-1)

    def score_embedding_candidates(
        self, users: Tensor, item_embeddings: Tensor
    ) -> Tensor:
        """Score a [batch, candidates, embedding] continuous candidate pool."""

        if item_embeddings.ndim != 3:
            raise ValueError("continuous candidate embeddings must be a tensor of rank 3")
        expected = (users.numel(), item_embeddings.shape[1], self.embedding_dim)
        if item_embeddings.shape != expected:
            raise ValueError(
                "continuous candidate embeddings must have shape "
                f"(batch, candidates, {self.embedding_dim})"
            )
        user = self.user_embedding(users) * self.h
        return torch.bmm(item_embeddings, user.unsqueeze(-1)).squeeze(-1)

    def all_scores(self, users: Tensor) -> Tensor:
        user = self.user_embedding(users) * self.h
        return user @ self.item_embedding.weight.T

    def feature(self, users: Tensor, items: Tensor) -> Tensor:
        return self.user_embedding(users) * self.item_embedding(items)

    def selected_l2(
        self,
        users: Tensor,
        *item_groups: Tensor,
        include_weights: bool = True,
    ) -> Tensor:
        result = self.user_embedding(users).square().sum()
        for items in item_groups:
            result = result + self.item_embedding(items).square().sum()
        return 0.5 * result


class MLP(Recommender):
    """MLP architecture matching ``baseline_model.py`` including layer_num=0."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        layer_count: int = 1,
    ) -> None:
        super().__init__()
        if layer_count < 0:
            raise ValueError("layer_count must be non-negative")
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.layer_count = layer_count
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        layers: list[nn.Linear] = []
        input_dim = 2 * embedding_dim
        for _ in range(layer_count):
            output_dim = input_dim // 2
            layers.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim
        self.layers = nn.ModuleList(layers)
        self.h = nn.Parameter(torch.empty(input_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        for index, layer in enumerate(self.layers):
            fan_in, fan_out = layer.in_features, layer.out_features
            # The legacy code uses a special (wider) bound for its one-layer MLP.
            bound = (
                math.sqrt(6.0 / fan_out)
                if self.layer_count == 1 and index == 0
                else math.sqrt(6.0 / (fan_in + fan_out))
            )
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)
        bound = math.sqrt(3.0 / self.h.numel())
        nn.init.uniform_(self.h, -bound, bound)

    def _feature_from_embeddings(self, user: Tensor, item: Tensor) -> Tensor:
        value = torch.cat((user, item), dim=-1)
        for layer in self.layers:
            value = torch.relu(layer(value))
        return value

    def score(self, users: Tensor, items: Tensor) -> Tensor:
        feature = self._feature_from_embeddings(
            self.user_embedding(users), self.item_embedding(items)
        )
        return (feature * self.h).sum(dim=-1)

    def score_candidates(self, users: Tensor, candidates: Tensor) -> Tensor:
        user = (
            self.user_embedding(users).unsqueeze(1).expand(-1, candidates.shape[1], -1)
        )
        item = self.item_embedding(candidates)
        feature = self._feature_from_embeddings(user, item)
        return (feature * self.h).sum(dim=-1)

    def all_scores(self, users: Tensor) -> Tensor:
        batch = users.shape[0]
        user = self.user_embedding(users).unsqueeze(1).expand(-1, self.num_items, -1)
        item = self.item_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        feature = self._feature_from_embeddings(user, item)
        return (feature * self.h).sum(dim=-1)

    def feature(self, users: Tensor, items: Tensor) -> Tensor:
        return self._feature_from_embeddings(
            self.user_embedding(users), self.item_embedding(items)
        )

    def selected_l2(
        self,
        users: Tensor,
        *item_groups: Tensor,
        include_weights: bool = True,
    ) -> Tensor:
        result = self.user_embedding(users).square().sum()
        for items in item_groups:
            result = result + self.item_embedding(items).square().sum()
        if include_weights:
            for layer in self.layers:
                result = result + layer.weight.square().sum()
        return 0.5 * result


class ItemPopularity(Recommender):
    def __init__(self, item_counts: Tensor, num_users: int) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = int(item_counts.numel())
        self.embedding_dim = 0
        self.register_buffer("item_counts", item_counts.float().reshape(-1))

    def score(self, users: Tensor, items: Tensor) -> Tensor:
        return self.item_counts[items]

    def score_candidates(self, users: Tensor, candidates: Tensor) -> Tensor:
        return self.item_counts[candidates]

    def all_scores(self, users: Tensor) -> Tensor:
        return self.item_counts.unsqueeze(0).expand(users.shape[0], -1)

    def feature(self, users: Tensor, items: Tensor) -> Tensor:
        return self.item_counts[items].unsqueeze(-1)

    def selected_l2(
        self,
        users: Tensor,
        *item_groups: Tensor,
        include_weights: bool = True,
    ) -> Tensor:
        return self.item_counts.new_zeros(())


def build_model(
    architecture: str,
    num_users: int,
    num_items: int,
    embedding_dim: int,
    mlp_layers: int = 0,
) -> Recommender:
    architecture = architecture.lower()
    if architecture == "gmf":
        return GMF(num_users, num_items, embedding_dim)
    if architecture == "mlp":
        return MLP(num_users, num_items, embedding_dim, mlp_layers)
    raise ValueError(f"Unknown architecture: {architecture}")


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameters_l2(parameters: Iterable[Tensor]) -> Tensor:
    values = list(parameters)
    if not values:
        return torch.tensor(0.0)
    return 0.5 * sum(
        (value.square().sum() for value in values), start=values[0].new_zeros(())
    )
