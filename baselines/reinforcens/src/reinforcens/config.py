from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    """Training options for the PyTorch ReinforceNS baselines."""

    model: str = "rns"
    architecture: str = "gmf"
    num_users: int = 16_015
    num_items: int = 45_782
    embedding_dim: int = 32
    mlp_layers: int = 0
    learning_rate: float = 1e-3
    generator_learning_rate: float | None = None
    optimizer: str = "adam"
    discriminator_reg: float = 1e-5
    generator_reg: float = 1e-5
    batch_size: int = 1_024
    epochs: int = 400
    num_negatives: int = 1
    candidates: int = 30
    exposure_candidates: int = 1
    dns_candidates: int = 50
    temperature: float = 1.0
    alpha: float = 2.5
    beta: float = 0.75
    sigma_range: tuple[float, ...] = (19.0, 20.0, 21.0, 22.0, 23.0)
    entropy_target: float = 1.0
    reduced: bool = True
    no_dns_loss: bool = False
    freeze_generator: bool = False
    exposure_pretrain_ratio: float = 0.0
    generator_pretrain_epochs: int = 0
    generator_pretrain_batch_size: int = 16_384
    generator_pretrain_negatives: int = 16
    generator_validation_fraction: float = 0.05
    generator_validation_examples: int = 131_072
    eval_mode: str = "list"
    recommendation_list_length: int = 160
    top_k: int = 100
    eval_batch_size: int = 1_024
    eval_every: int = 1
    early_stopping_patience: int = 10
    select_by: str = "auc"
    seed: int = 1
    device: str = "auto"
    amp: bool = False
    compile_model: bool = False
    drop_last: bool = True
    clean_exposure_conflicts: bool = True
    max_train_examples: int | None = None

    def __post_init__(self) -> None:
        self.model = self.model.lower()
        self.architecture = self.architecture.lower()
        self.optimizer = self.optimizer.lower()
        self.eval_mode = self.eval_mode.lower()
        self.select_by = self.select_by.lower()
        if self.model not in {
            "bpr",
            "dns",
            "kbgan",
            "rns",
            "eprns",
            "berns",
            "cberns",
            "ehrns",
            "itempop",
        }:
            raise ValueError(f"Unsupported model: {self.model}")
        if self.architecture not in {"gmf", "mlp"}:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        if self.optimizer not in {"adam", "adagrad", "sgd"}:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        if self.eval_mode not in {"list", "topk"}:
            raise ValueError(f"Unsupported eval mode: {self.eval_mode}")
        if not 0 <= self.exposure_candidates <= self.candidates:
            raise ValueError("exposure_candidates must be between 0 and candidates")
        if self.candidates <= 0 or self.dns_candidates <= 0 or self.num_negatives <= 0:
            raise ValueError("candidate and negative counts must be positive")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if not 0.0 <= self.exposure_pretrain_ratio <= 1.0:
            raise ValueError("exposure_pretrain_ratio must be in [0, 1]")
        if self.generator_pretrain_epochs < 0:
            raise ValueError("generator pretrain epochs must be non-negative")
        if self.generator_pretrain_batch_size <= 0:
            raise ValueError("generator pretrain batch size must be positive")
        if self.generator_pretrain_negatives <= 0:
            raise ValueError("generator pretrain negatives must be positive")
        if not 0 < self.generator_validation_fraction < 0.5:
            raise ValueError("generator validation fraction must be in (0, 0.5)")
        if self.generator_validation_examples <= 0:
            raise ValueError("generator validation examples must be positive")
        if self.batch_size <= 0 or self.epochs < 0:
            raise ValueError("batch_size must be positive and epochs non-negative")
        if self.temperature <= 0 or self.entropy_target <= 0:
            raise ValueError("temperature and entropy_target must be positive")
        if self.model == "ehrns" and not self.reduced:
            raise ValueError("EH-RNS currently requires reduced candidate sampling")

    @property
    def generator_lr(self) -> float:
        return (
            self.learning_rate
            if self.generator_learning_rate is None
            else self.generator_learning_rate
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        values = dict(values)
        if "sigma_range" in values:
            values["sigma_range"] = tuple(values["sigma_range"])
        return cls(**values)
