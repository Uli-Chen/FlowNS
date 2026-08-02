from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .checkpoint import load_legacy_weights, save_checkpoint
from .config import TrainConfig
from .data import EvalCandidates, InteractionData
from .metrics import EvalResult, evaluate_list, evaluate_topk, format_result
from .models import (
    BarycentricGenerator,
    ItemPopularity,
    Recommender,
    build_model,
    exposure_calibrated_hard_tilt,
)
from .sampling import NegativeSampler


RNS_METHODS = {"rns", "eprns", "berns", "cberns", "ehrns"}
ADVERSARIAL_METHODS = {"kbgan", *RNS_METHODS}


@dataclass(slots=True)
class EpochStats:
    epoch: int
    discriminator_loss: float
    generator_loss: float
    hard_reward: float
    exposure_reward: float
    exact_exposure_rate: float
    examples: int
    seconds: float
    examples_per_second: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "epoch": self.epoch,
            "discriminator_loss": self.discriminator_loss,
            "generator_loss": self.generator_loss,
            "hard_reward": self.hard_reward,
            "exposure_reward": self.exposure_reward,
            "exact_exposure_rate": self.exact_exposure_rate,
            "examples": self.examples,
            "train_seconds": self.seconds,
            "train_examples_per_second": self.examples_per_second,
        }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def negative_mmd_reward(
    real_features: Tensor,
    fake_features: Tensor,
    sigma_range: tuple[float, ...],
) -> tuple[Tensor, Tensor]:
    """Exact multi-kernel MMD reward from ``Discriminator._compute_MMD``."""

    x_sq = real_features.square().sum(dim=1, keepdim=True)
    y_sq = fake_features.square().sum(dim=1, keepdim=True)
    xx = -2.0 * (real_features @ real_features.T) + x_sq + x_sq.T
    xy = -2.0 * (real_features @ fake_features.T) + x_sq + y_sq.T
    yy = -2.0 * (fake_features @ fake_features.T) + y_sq + y_sq.T
    kxx = real_features.new_zeros(())
    kxy = real_features.new_zeros(())
    kyy = real_features.new_zeros(())
    kxy_array = fake_features.new_zeros(fake_features.shape[0])
    kyy_array = fake_features.new_zeros(fake_features.shape[0])
    for sigma in sigma_range:
        denominator = 2.0 * sigma**2
        kernel_xx = torch.exp(-xx / denominator)
        kernel_xy = torch.exp(-xy / denominator)
        kernel_yy = torch.exp(-yy / denominator)
        kxx = kxx + kernel_xx.mean()
        kxy = kxy + kernel_xy.mean()
        kyy = kyy + kernel_yy.mean()
        kxy_array = kxy_array + kernel_xy.mean(dim=0)
        kyy_array = kyy_array + kernel_yy.mean(dim=0)
    per_sample = -(2.0 * kyy_array - 2.0 * kxy_array)
    global_reward = -torch.sqrt((kxx + kyy - 2.0 * kxy).clamp_min(0.0))
    return global_reward, per_sample


class Trainer:
    def __init__(
        self,
        data: InteractionData,
        config: TrainConfig,
        *,
        pretrained_checkpoint: str | Path | None = None,
    ) -> None:
        self.data = data
        self.config = config
        if (config.num_users, config.num_items) != (data.num_users, data.num_items):
            raise ValueError(
                "configuration user/item dimensions do not match the loaded dataset"
            )
        self.device = resolve_device(config.device)
        seed_everything(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.sampler = NegativeSampler(data, config.seed + 1)
        self.eval_candidates: dict[str, EvalCandidates] = {}
        self.amp_enabled = bool(config.amp and self.device.type == "cuda")
        if config.amp and self.device.type != "cuda":
            raise ValueError("--amp currently requires a CUDA device")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        if config.model == "itempop":
            counts = np.bincount(data.train_items, minlength=data.num_items)
            self.discriminator: Recommender = ItemPopularity(
                torch.from_numpy(counts), data.num_users
            ).to(self.device)
            self.generator = None
            self.discriminator_optimizer = None
            self.generator_optimizer = None
            return

        self.discriminator = build_model(
            config.architecture,
            data.num_users,
            data.num_items,
            config.embedding_dim,
            config.mlp_layers,
        ).to(self.device)
        self.generator: torch.nn.Module | None = None
        if config.model in ADVERSARIAL_METHODS:
            policy = build_model(
                config.architecture,
                data.num_users,
                data.num_items,
                config.embedding_dim,
                config.mlp_layers,
            ).to(self.device)
            if config.model in {"berns", "cberns", "ehrns"}:
                rng_after_policy = torch.random.get_rng_state()
                exposure_prior = build_model(
                    config.architecture,
                    data.num_users,
                    data.num_items,
                    config.embedding_dim,
                    config.mlp_layers,
                ).to(self.device)
                torch.random.set_rng_state(rng_after_policy)
                exposure_prior.requires_grad_(False)
                self.generator = BarycentricGenerator(
                    policy, exposure_prior
                ).to(self.device)
            else:
                self.generator = policy
        if pretrained_checkpoint is not None:
            load_legacy_weights(self.discriminator, pretrained_checkpoint)
            if config.model in ADVERSARIAL_METHODS and self.generator is not None:
                if isinstance(self.generator, BarycentricGenerator):
                    load_legacy_weights(self.generator.policy, pretrained_checkpoint)
                    load_legacy_weights(
                        self.generator.exposure_prior, pretrained_checkpoint
                    )
                else:
                    load_legacy_weights(self.generator, pretrained_checkpoint)
        if config.compile_model and hasattr(torch, "compile"):
            self.discriminator = torch.compile(self.discriminator)  # type: ignore[assignment]
            if self.generator is not None:
                self.generator = torch.compile(self.generator)  # type: ignore[assignment]
        self.discriminator_optimizer = self._make_optimizer(
            self.discriminator.parameters(), config.learning_rate
        )
        self.generator_optimizer = (
            None
            if self.generator is None
            else self._make_optimizer(
                (
                    self.generator.policy.parameters()
                    if isinstance(self.generator, BarycentricGenerator)
                    else self.generator.parameters()
                ),
                config.generator_lr,
            )
        )

    def _make_optimizer(
        self, parameters: Any, learning_rate: float
    ) -> torch.optim.Optimizer:
        if self.config.optimizer == "adam":
            return torch.optim.Adam(parameters, lr=learning_rate)
        if self.config.optimizer == "adagrad":
            return torch.optim.Adagrad(
                parameters, lr=learning_rate, initial_accumulator_value=1e-8
            )
        return torch.optim.SGD(parameters, lr=learning_rate)

    def _batch_indices(self) -> np.ndarray:
        size = self.data.train_users.size
        maximum = self.config.max_train_examples
        if maximum is not None and maximum < size:
            return self.rng.choice(size, size=maximum, replace=False)
        return self.rng.permutation(size)

    def _iter_batches(self):
        indices = self._batch_indices()
        batch_size = self.config.batch_size
        stop = (
            len(indices)
            if not self.config.drop_last
            else len(indices) // batch_size * batch_size
        )
        for begin in range(0, stop, batch_size):
            selected = indices[begin : min(begin + batch_size, stop)]
            yield self.data.train_users[selected], self.data.train_items[selected]

    def _tensor(self, values: np.ndarray) -> Tensor:
        return torch.from_numpy(values.astype(np.int64, copy=False)).to(
            self.device, non_blocking=True
        )

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _step(self, loss: Tensor, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

    def train_epoch(
        self, epoch: int, hard_baseline: float = 0.0, exposure_baseline: float = 0.0
    ) -> EpochStats:
        if self.config.model == "itempop":
            return EpochStats(epoch, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
        self.discriminator.train()
        if self.generator is not None:
            if isinstance(self.generator, BarycentricGenerator):
                prior_weight = (
                    1.0 / (epoch + 1)
                    if self.config.model in {"cberns", "ehrns"}
                    else 0.5
                )
                self.generator.set_prior_weight(prior_weight)
            self.generator.train()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        totals = {
            "dis": 0.0,
            "gen": 0.0,
            "hard": 0.0,
            "exposure": 0.0,
            "exact": 0.0,
            "examples": 0,
        }
        for users_np, positives_np in self._iter_batches():
            if self.config.model == "bpr":
                self._train_bpr_batch(users_np, positives_np, totals)
            elif self.config.model == "dns":
                self._train_dns_batch(users_np, positives_np, totals)
            else:
                self._train_adversarial_batch(
                    users_np,
                    positives_np,
                    totals,
                    hard_baseline,
                    exposure_baseline,
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - start
        examples = int(totals["examples"])
        denominator = max(examples, 1)
        return EpochStats(
            epoch=epoch,
            discriminator_loss=totals["dis"] / denominator,
            generator_loss=totals["gen"] / denominator,
            hard_reward=totals["hard"] / denominator,
            exposure_reward=totals["exposure"] / denominator,
            exact_exposure_rate=totals["exact"] / denominator,
            examples=examples,
            seconds=seconds,
            examples_per_second=examples / max(seconds, 1e-12),
        )

    def _train_bpr_batch(
        self, users_np: np.ndarray, positives_np: np.ndarray, totals: dict[str, float]
    ) -> None:
        count = self.config.num_negatives
        negatives_np = self.sampler.bpr(
            users_np, count, self.config.exposure_pretrain_ratio
        )
        users_np = np.repeat(users_np, count)
        positives_np = np.repeat(positives_np, count)
        negatives_np = negatives_np.reshape(-1)
        users, positives, negatives = map(
            self._tensor, (users_np, positives_np, negatives_np)
        )
        with self._autocast():
            positive_scores = self.discriminator.score(users, positives)
            negative_scores = self.discriminator.score(users, negatives)
            loss = F.softplus(-(positive_scores - negative_scores)).sum()
            if self.config.discriminator_reg:
                loss = (
                    loss
                    + self.config.discriminator_reg
                    * self.discriminator.selected_l2(users, positives, negatives)
                )
        assert self.discriminator_optimizer is not None
        self._step(loss, self.discriminator_optimizer)
        totals["dis"] += float(loss.detach())
        totals["examples"] += int(users.numel())

    def _train_dns_batch(
        self, users_np: np.ndarray, positives_np: np.ndarray, totals: dict[str, float]
    ) -> None:
        candidates_np = self.sampler.uniform(users_np, self.config.dns_candidates)
        users = self._tensor(users_np)
        positives = self._tensor(positives_np)
        candidates = self._tensor(candidates_np)
        with torch.no_grad():
            with self._autocast():
                candidate_scores = self.discriminator.score_candidates(
                    users, candidates
                )
            negatives = candidates.gather(
                1, candidate_scores.argmax(dim=1, keepdim=True)
            ).squeeze(1)
        with self._autocast():
            positive_scores = self.discriminator.score(users, positives)
            negative_scores = self.discriminator.score(users, negatives)
            loss = F.softplus(-(positive_scores - negative_scores)).sum()
            if self.config.discriminator_reg:
                loss = (
                    loss
                    + self.config.discriminator_reg
                    * self.discriminator.selected_l2(users, positives, negatives)
                )
        assert self.discriminator_optimizer is not None
        self._step(loss, self.discriminator_optimizer)
        totals["dis"] += float(loss.detach())
        totals["examples"] += int(users.numel())

    def _sample_generator(
        self, users_np: np.ndarray, users: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None]:
        assert self.generator is not None
        if self.config.reduced:
            if self.config.model in RNS_METHODS:
                candidates_np = self.sampler.rns_candidates(
                    users_np, self.config.candidates, self.config.exposure_candidates
                )
            else:
                candidates_np = self.sampler.kbgan_candidates(
                    users_np, self.config.candidates
                )
            candidates = self._tensor(candidates_np)
            with torch.no_grad():
                with self._autocast():
                    logits = (
                        self.generator.score_candidates(users, candidates)
                        / self.config.temperature
                    )
                sampling_offset = None
                if self.config.model == "ehrns":
                    if not isinstance(self.generator, BarycentricGenerator):
                        raise TypeError("EH-RNS requires a barycentric generator")
                    ranker_scores = self.discriminator.score_candidates(
                        users, candidates
                    )
                    density_logits = self.generator.exposure_prior.score_candidates(
                        users, candidates
                    )
                    sampling_offset = exposure_calibrated_hard_tilt(
                        ranker_scores, density_logits
                    )
                    logits = logits.float() + sampling_offset
                action_ids = torch.multinomial(torch.softmax(logits, dim=1), 1)
                negatives = candidates.gather(1, action_ids).squeeze(1)
            return negatives, candidates, action_ids, sampling_offset

        with torch.no_grad():
            with self._autocast():
                logits = self.generator.all_scores(users) / self.config.temperature
            for row, user_np in enumerate(users_np):
                clicked = self.data.train_clicks.for_user(int(user_np))
                if clicked.size:
                    logits[row, self._tensor(clicked)] = -torch.inf
            action_ids = torch.multinomial(torch.softmax(logits, dim=1), 1)
        return action_ids.squeeze(1), None, action_ids, None

    def _train_adversarial_batch(
        self,
        users_np: np.ndarray,
        positives_np: np.ndarray,
        totals: dict[str, float],
        hard_baseline: float,
        exposure_baseline: float,
    ) -> None:
        assert self.generator is not None and self.generator_optimizer is not None
        users = self._tensor(users_np)
        positives = self._tensor(positives_np)
        negatives, candidates, action_ids, sampling_offset = self._sample_generator(
            users_np, users
        )

        with self._autocast():
            positive_scores = self.discriminator.score(users, positives)
            negative_scores = self.discriminator.score(users, negatives)
            hard_reward = -torch.sigmoid(-negative_scores.detach()).float()
            discriminator_loss = F.softplus(-(positive_scores - negative_scores)).sum()
            if self.config.discriminator_reg:
                discriminator_loss = (
                    discriminator_loss
                    + self.config.discriminator_reg
                    * self.discriminator.selected_l2(users, positives, negatives)
                )
        assert self.discriminator_optimizer is not None
        self._step(discriminator_loss, self.discriminator_optimizer)

        if self.config.model in RNS_METHODS:
            real_exposure_np = self.sampler.exposed(users_np, 1).reshape(-1)
            real_exposure = self._tensor(real_exposure_np)
            with torch.no_grad():
                real_features = self.discriminator.feature(users, real_exposure).float()
                fake_features = self.discriminator.feature(users, negatives).float()
                _, feature_reward = negative_mmd_reward(
                    real_features, fake_features, self.config.sigma_range
                )
            negative_np = negatives.detach().cpu().numpy().astype(np.int32, copy=False)
            exact_np = self.sampler.is_exposed(users_np, negative_np).astype(np.float32)
            exact_reward = torch.from_numpy(exact_np).to(self.device)
            exposure_reward = (
                1.0 - self.config.beta
            ) * exact_reward + self.config.beta * feature_reward
        else:
            exact_np = self.sampler.is_exposed(
                users_np, negatives.detach().cpu().numpy().astype(np.int32, copy=False)
            ).astype(np.float32)
            exposure_reward = torch.zeros_like(hard_reward)

        generator_loss_value = 0.0
        if not self.config.freeze_generator:
            with self._autocast():
                if candidates is None:
                    logits = self.generator.all_scores(users) / self.config.temperature
                    for row, user_np in enumerate(users_np):
                        clicked = self.data.train_clicks.for_user(int(user_np))
                        if clicked.size:
                            logits[row, self._tensor(clicked)] = -torch.inf
                else:
                    logits = (
                        self.generator.score_candidates(users, candidates)
                        / self.config.temperature
                    )
                    if sampling_offset is not None:
                        logits = logits.float() + sampling_offset
                log_probability = (
                    torch.log_softmax(logits, dim=1).gather(1, action_ids).squeeze(1)
                )
                if self.config.no_dns_loss and self.config.model in RNS_METHODS:
                    advantage = exposure_reward - exposure_baseline
                else:
                    advantage = (hard_reward - hard_baseline) + self.config.alpha * (
                        exposure_reward - exposure_baseline
                    )
                generator_loss = -(log_probability * advantage.detach()).sum()
                if not self.config.reduced:
                    entropy = -(
                        torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1)
                    ).sum(dim=1)
                    target = math.log(self.config.entropy_target)
                    generator_loss = (
                        generator_loss
                        + torch.minimum(
                            torch.zeros_like(entropy),
                            entropy.new_full(entropy.shape, target) - entropy,
                        ).sum()
                    )
                if self.config.generator_reg:
                    generator_loss = (
                        generator_loss
                        + self.config.generator_reg
                        * self.generator.selected_l2(users, negatives)
                    )
            self._step(generator_loss, self.generator_optimizer)
            generator_loss_value = float(generator_loss.detach())

        batch_size = int(users.numel())
        totals["dis"] += float(discriminator_loss.detach())
        totals["gen"] += generator_loss_value
        totals["hard"] += float(hard_reward.sum())
        totals["exposure"] += float(exposure_reward.sum())
        totals["exact"] += float(exact_np.sum())
        totals["examples"] += batch_size

    def evaluate(self, split: str = "validation") -> EvalResult:
        if self.config.eval_mode == "list":
            if split not in self.eval_candidates:
                self.eval_candidates[split] = self.data.make_eval_candidates(
                    split,
                    self.config.recommendation_list_length,
                    self.config.seed,
                )
            return evaluate_list(
                self.discriminator,
                self.eval_candidates[split],
                device=self.device,
                batch_size=self.config.eval_batch_size,
            )
        return evaluate_topk(
            self.discriminator,
            self.data,
            split=split,
            k=self.config.top_k,
            device=self.device,
            batch_size=min(self.config.eval_batch_size, 256),
        )

    def pretrain_generator_density(self) -> list[dict[str, float | int]]:
        """Fit a generator or frozen prior as a train-exposure density ratio."""

        if self.config.model not in {
            "eprns",
            "berns",
            "cberns",
            "ehrns",
        }:
            return []
        if self.generator is None or self.generator_optimizer is None:
            raise RuntimeError("exposure-density pretraining requires a generator")
        density_model = (
            self.generator.exposure_prior
            if isinstance(self.generator, BarycentricGenerator)
            else self.generator
        )
        if self.config.generator_pretrain_epochs == 0:
            density_model.requires_grad_(False)
            return []
        density_model.requires_grad_(True)
        density_optimizer = self._make_optimizer(
            density_model.parameters(), self.config.generator_lr
        )
        exposure_keys = self.data.train_exposures.keys
        if exposure_keys.size < 2:
            raise ValueError("density pretraining requires at least two exposure pairs")

        split_rng = np.random.default_rng(self.config.seed + 73)
        shuffled_keys = split_rng.permutation(exposure_keys)
        requested_validation = int(
            exposure_keys.size * self.config.generator_validation_fraction
        )
        validation_count = min(
            max(requested_validation, 1),
            self.config.generator_validation_examples,
            exposure_keys.size - 1,
        )
        validation_keys = shuffled_keys[:validation_count]
        training_keys = shuffled_keys[validation_count:]
        validation_users_np = (validation_keys // self.data.num_items).astype(
            np.int32, copy=False
        )
        validation_items_np = (validation_keys % self.data.num_items).astype(
            np.int32, copy=False
        )
        validation_sampler = NegativeSampler(self.data, self.config.seed + 79)
        validation_negatives_np = validation_sampler.uniform(
            validation_users_np, self.config.generator_pretrain_negatives
        )
        train_sampler = NegativeSampler(self.data, self.config.seed + 83)
        train_rng = np.random.default_rng(self.config.seed + 89)
        best_validation = float("inf")
        best_state: dict[str, Tensor] | None = None
        records: list[dict[str, float | int]] = []
        density_model.train()

        for epoch in range(1, self.config.generator_pretrain_epochs + 1):
            order = train_rng.permutation(training_keys.size)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            loss_total = 0.0
            processed = 0
            batch_size = self.config.generator_pretrain_batch_size
            for begin in range(0, training_keys.size, batch_size):
                selected_keys = training_keys[
                    order[begin : min(begin + batch_size, training_keys.size)]
                ]
                users_np = (selected_keys // self.data.num_items).astype(
                    np.int32, copy=False
                )
                positives_np = (selected_keys % self.data.num_items).astype(
                    np.int32, copy=False
                )
                negatives_np = train_sampler.uniform(
                    users_np, self.config.generator_pretrain_negatives
                )
                users = self._tensor(users_np)
                positives = self._tensor(positives_np)
                negatives = self._tensor(negatives_np)
                with self._autocast():
                    positive_logits = density_model.score(users, positives)
                    negative_logits = density_model.score_candidates(users, negatives)
                    loss = F.softplus(-positive_logits).mean() + F.softplus(
                        negative_logits
                    ).mean()
                    if self.config.generator_reg:
                        loss = (
                            loss
                            + self.config.generator_reg
                            * density_model.selected_l2(users, positives, negatives)
                            / users.numel()
                        )
                self._step(loss, density_optimizer)
                count = int(users.numel())
                loss_total += float(loss.detach()) * count
                processed += count

            validation_loss, validation_auc = self._evaluate_generator_density(
                validation_users_np,
                validation_items_np,
                validation_negatives_np,
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                generator = getattr(density_model, "_orig_mod", density_model)
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in generator.state_dict().items()
                }
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            seconds = time.perf_counter() - start
            record: dict[str, float | int] = {
                "epoch": epoch,
                "nce_loss": loss_total / max(processed, 1),
                "validation_nce_loss": validation_loss,
                "validation_auc": validation_auc,
                "training_exposures": int(training_keys.size),
                "validation_exposures": int(validation_keys.size),
                "processed_examples": processed,
                "train_seconds": seconds,
                "train_examples_per_second": processed / max(seconds, 1e-12),
            }
            records.append(record)
            print(
                f"Generator pretrain {epoch}: nce={record['nce_loss']:.6f} "
                f"val_nce={validation_loss:.6f} val_auc={validation_auc:.4f} "
                f"{record['train_examples_per_second']:,.0f} exposures/s",
                flush=True,
            )

        if best_state is None:
            raise RuntimeError("generator density pretraining produced no checkpoint")
        generator = getattr(density_model, "_orig_mod", density_model)
        generator.load_state_dict(best_state)
        density_model.requires_grad_(False)
        self.generator_optimizer = self._make_optimizer(
            (
                self.generator.policy.parameters()
                if isinstance(self.generator, BarycentricGenerator)
                else self.generator.parameters()
            ),
            self.config.generator_lr,
        )
        self.generator.train()
        return records

    @torch.inference_mode()
    def _evaluate_generator_density(
        self,
        users_np: np.ndarray,
        positives_np: np.ndarray,
        negatives_np: np.ndarray,
    ) -> tuple[float, float]:
        if self.generator is None:
            raise RuntimeError("generator density evaluation requires a generator")
        density_model = (
            self.generator.exposure_prior
            if isinstance(self.generator, BarycentricGenerator)
            else self.generator
        )
        density_model.eval()
        loss_total = 0.0
        correct_total = 0.0
        count_total = 0
        batch_size = self.config.generator_pretrain_batch_size
        for begin in range(0, users_np.size, batch_size):
            end = min(begin + batch_size, users_np.size)
            users = self._tensor(users_np[begin:end])
            positives = self._tensor(positives_np[begin:end])
            negatives = self._tensor(negatives_np[begin:end])
            positive_logits = density_model.score(users, positives).float()
            negative_logits = density_model.score_candidates(users, negatives).float()
            batch_loss = F.softplus(-positive_logits) + F.softplus(
                negative_logits
            ).mean(dim=1)
            loss_total += float(batch_loss.sum())
            correct_total += float(
                (positive_logits[:, None] > negative_logits).float().sum()
            )
            count_total += int(users.numel())
        density_model.train()
        pair_count = count_total * negatives_np.shape[1]
        return loss_total / max(count_total, 1), correct_total / max(pair_count, 1)

    def fit(self, output_dir: str | Path) -> list[dict[str, Any]]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pretrain_history = self.pretrain_generator_density()
        if pretrain_history:
            (output / "generator_pretrain_history.json").write_text(
                json.dumps(pretrain_history, indent=2) + "\n", encoding="utf-8"
            )
        history: list[dict[str, Any]] = []
        hard_baseline = 0.0
        exposure_baseline = 0.0
        best_metric = -float("inf")
        stale = 0

        initial = self.evaluate("validation")
        print(f"Initial validation: {format_result(initial)}", flush=True)
        initial_record = {
            "epoch": 0,
            **{f"validation_{k}": v for k, v in initial.to_dict().items()},
        }
        if pretrain_history:
            initial_record.update(
                {
                    "generator_pretrain_epochs": len(pretrain_history),
                    "generator_pretrain_seconds": sum(
                        float(record["train_seconds"]) for record in pretrain_history
                    ),
                    "generator_pretrain_final_loss": pretrain_history[-1]["nce_loss"],
                }
            )
        history.append(initial_record)
        best_metric = self._selection_metric(initial)
        save_checkpoint(
            output / "best.pt",
            config=self.config,
            discriminator=self.discriminator,
            generator=self.generator,
            epoch=0,
            metrics=initial.to_dict(),
            history=history,
        )

        for epoch in range(1, self.config.epochs + 1):
            stats = self.train_epoch(epoch, hard_baseline, exposure_baseline)
            if self.config.model in ADVERSARIAL_METHODS:
                hard_baseline = stats.hard_reward
                exposure_baseline = stats.exposure_reward
            record: dict[str, Any] = stats.to_dict()
            print(
                f"Epoch {epoch}: dis={stats.discriminator_loss:.6f} gen={stats.generator_loss:.6f} "
                f"hard={stats.hard_reward:.6f} exposure={stats.exposure_reward:.6f} "
                f"exact={stats.exact_exposure_rate:.4f} "
                f"{stats.examples_per_second:,.0f} examples/s",
                flush=True,
            )
            if epoch % self.config.eval_every == 0:
                result = self.evaluate("validation")
                print(f"Validation: {format_result(result)}", flush=True)
                record.update(
                    {f"validation_{k}": v for k, v in result.to_dict().items()}
                )
                metric = self._selection_metric(result)
                if metric > best_metric:
                    best_metric = metric
                    stale = 0
                    save_checkpoint(
                        output / "best.pt",
                        config=self.config,
                        discriminator=self.discriminator,
                        generator=self.generator,
                        epoch=epoch,
                        metrics=result.to_dict(),
                        history=history + [record],
                    )
                else:
                    stale += 1
            history.append(record)
            (output / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            save_checkpoint(
                output / "last.pt",
                config=self.config,
                discriminator=self.discriminator,
                generator=self.generator,
                epoch=epoch,
                metrics={
                    key: value
                    for key, value in record.items()
                    if key.startswith("validation_")
                },
                history=history,
            )
            if stale > self.config.early_stopping_patience:
                print(
                    f"Early stopping after {stale} non-improving evaluations",
                    flush=True,
                )
                break
        return history

    def _selection_metric(self, result: EvalResult) -> float:
        return result.ndcg if self.config.select_by == "ndcg" else result.primary
