from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from reinforcens.checkpoint import build_from_legacy
from reinforcens.data import InteractionData
from reinforcens.models import GMF
from reinforcens.trainer import resolve_device

from .flow import ExposureFlowConfig
from .flow_audit import ExposureFlowAuditConfig, audit_exposure_flow
from .flow_ranker import ContinuousFlowRankerTrainer, ContinuousRankerConfig
from .flow_training import (
    ExposureFlowTrainer,
    FlowTrainingConfig,
    load_exposure_flow,
    split_train_exposures,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = PROJECT_ROOT.parent / "baselines" / "reinforcens"
DEFAULT_DATA = BASELINE_ROOT / "data" / "zhihu" / "train-valid-test.zip"
DEFAULT_PRETRAINED = (
    BASELINE_ROOT / "checkpoints" / "pretrain_model_dis_zhihu.pkl"
)
DEFAULT_CACHE = BASELINE_ROOT / ".cache" / "zhihu.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and inspect a train-only conditional exposure flow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pretrain = subparsers.add_parser("pretrain")
    pretrain.add_argument("--data", type=Path, default=DEFAULT_DATA)
    pretrain.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    pretrain.add_argument(
        "--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED
    )
    pretrain.add_argument("--initialize-from", type=Path)
    pretrain.add_argument("--output", type=Path, required=True)
    pretrain.add_argument("--hidden-dim", type=int, default=512)
    pretrain.add_argument("--depth", type=int, default=8)
    pretrain.add_argument("--user-dim", type=int, default=128)
    pretrain.add_argument("--time-dim", type=int, default=128)
    pretrain.add_argument("--sampling-steps", type=int, default=16)
    pretrain.add_argument("--reconstruction-weight", type=float, default=0.25)
    pretrain.add_argument("--batch-size", type=int, default=2_048)
    pretrain.add_argument("--learning-rate", type=float, default=2e-4)
    pretrain.add_argument("--weight-decay", type=float, default=1e-5)
    pretrain.add_argument("--epochs", type=int, default=50)
    pretrain.add_argument("--minimum-epochs", type=int, default=10)
    pretrain.add_argument("--early-stopping", type=int, default=5)
    pretrain.add_argument("--minimum-relative-improvement", type=float, default=0.005)
    pretrain.add_argument("--validation-fraction", type=float, default=0.02)
    pretrain.add_argument("--validation-examples", type=int, default=131_072)
    pretrain.add_argument("--diagnostics-examples", type=int, default=4_096)
    pretrain.add_argument("--diagnostics-projections", type=int, default=64)
    pretrain.add_argument("--maximum-training-exposures", type=int)
    pretrain.add_argument("--gradient-clip", type=float, default=1.0)
    pretrain.add_argument("--seed", type=int, default=1)
    pretrain.add_argument("--device", default="auto")
    pretrain.add_argument("--amp", action="store_true")

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    inspect.add_argument(
        "--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED
    )
    inspect.add_argument("--device", default="cpu")

    rank = subparsers.add_parser("rank-continuous")
    rank.add_argument("--data", type=Path, default=DEFAULT_DATA)
    rank.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    rank.add_argument(
        "--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED
    )
    rank.add_argument("--flow-checkpoint", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)
    rank.add_argument("--batch-size", type=int, default=2_048)
    rank.add_argument("--learning-rate", type=float, default=1e-3)
    rank.add_argument("--weight-decay", type=float, default=0.0)
    rank.add_argument("--discriminator-reg", type=float, default=1e-5)
    rank.add_argument("--epochs", type=int, default=30)
    rank.add_argument("--minimum-epochs", type=int, default=10)
    rank.add_argument("--early-stopping", type=int, default=6)
    rank.add_argument("--minimum-ndcg-improvement", type=float, default=1e-5)
    rank.add_argument("--candidates-per-user", type=int, default=16)
    rank.add_argument("--pool-batch-size", type=int, default=2_048)
    rank.add_argument("--pool-refresh-epochs", type=int, default=5)
    rank.add_argument("--flow-steps", type=int, default=16)
    rank.add_argument("--flow-solver", choices=("euler", "heun"), default="heun")
    rank.add_argument(
        "--negative-source",
        choices=("flow", "soft-flow", "flow-dns", "uniform"),
        default="flow",
    )
    rank.add_argument("--train-item-embeddings", action="store_true")
    rank.add_argument(
        "--selection-mode", choices=("random", "safe-hard"), default="random"
    )
    rank.add_argument("--hardness-beta", type=float, default=2.0)
    rank.add_argument("--minimum-ess-ratio", type=float, default=0.5)
    rank.add_argument("--safety-temperature", type=float, default=1.0)
    rank.add_argument("--transport-neighbors", type=int, default=8)
    rank.add_argument("--transport-minimum-ess-ratio", type=float, default=0.5)
    rank.add_argument(
        "--transport-max-inverse-temperature", type=float, default=20.0
    )
    rank.add_argument("--dns-uniform-candidates", type=int, default=29)
    rank.add_argument("--dns-flow-candidates", type=int, default=1)
    rank.add_argument("--ranking-loss", choices=("bpr", "hard-bpr"), default="bpr")
    rank.add_argument("--hard-bpr-a", type=float, default=1.0)
    rank.add_argument("--hard-bpr-b", type=float, default=-1.0)
    rank.add_argument("--hard-bpr-c", type=float, default=0.8)
    rank.add_argument("--recommendation-list-length", type=int, default=160)
    rank.add_argument("--eval-batch-size", type=int, default=1_024)
    rank.add_argument("--maximum-train-examples", type=int)
    rank.add_argument("--diagnostics-examples", type=int, default=1_024)
    rank.add_argument("--gradient-clip", type=float, default=0.0)
    rank.add_argument("--seed", type=int, default=1)
    rank.add_argument("--device", default="auto")
    rank.add_argument("--amp", action="store_true")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--data", type=Path, default=DEFAULT_DATA)
    audit.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    audit.add_argument(
        "--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED
    )
    audit.add_argument("--flow-checkpoint", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--examples", type=int, default=4_096)
    audit.add_argument("--nearest-examples", type=int, default=1_024)
    audit.add_argument("--projections", type=int, default=128)
    audit.add_argument(
        "--solver-steps", type=int, nargs="+", default=(4, 8, 16, 32)
    )
    audit.add_argument("--c2st-epochs", type=int, default=100)
    audit.add_argument("--c2st-hidden-dim", type=int, default=128)
    audit.add_argument("--seed", type=int, default=1)
    audit.add_argument("--device", default="auto")
    return parser


def _catalog(path: Path, device: str | torch.device) -> Tensor:
    model = build_from_legacy(path, device=device)
    if not isinstance(model, GMF):
        raise TypeError("exposure flow currently requires the released GMF geometry")
    return model.item_embedding.weight.detach()


def command_pretrain(args: argparse.Namespace) -> None:
    data = InteractionData.load(args.data, cache_path=args.cache)
    catalog = _catalog(args.pretrained_checkpoint, "cpu")
    flow_config = ExposureFlowConfig(
        num_users=data.num_users,
        embedding_dim=int(catalog.shape[1]),
        user_dim=args.user_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        sampling_steps=args.sampling_steps,
        reconstruction_weight=args.reconstruction_weight,
    )
    training_config = FlowTrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        early_stopping_patience=args.early_stopping,
        minimum_relative_improvement=args.minimum_relative_improvement,
        validation_fraction=args.validation_fraction,
        validation_examples=args.validation_examples,
        diagnostics_examples=args.diagnostics_examples,
        diagnostics_projections=args.diagnostics_projections,
        maximum_training_exposures=args.maximum_training_exposures,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        gradient_clip=args.gradient_clip,
    )
    trainer = ExposureFlowTrainer(data, catalog, flow_config, training_config)
    initialization = (
        None
        if args.initialize_from is None
        else trainer.initialize_from_checkpoint(args.initialize_from)
    )
    print(
        json.dumps(
            {
                "device": str(trainer.device),
                "parameters": trainer.flow.num_parameters,
                "train_exposures": int(trainer.training_keys.size),
                "holdout_exposures": int(trainer.validation_keys.size),
                "official_validation_or_test_used": False,
                "initialization": initialization,
            },
            indent=2,
        ),
        flush=True,
    )
    trainer.fit(args.output)


def command_inspect(args: argparse.Namespace) -> None:
    catalog = _catalog(args.pretrained_checkpoint, args.device)
    flow, metadata = load_exposure_flow(
        args.checkpoint, catalog, device=args.device
    )
    print(
        json.dumps(
            {
                "parameters": flow.num_parameters,
                "config": flow.config.to_dict(),
                "metadata": metadata,
            },
            indent=2,
        )
    )


def command_rank_continuous(args: argparse.Namespace) -> None:
    data = InteractionData.load(args.data, cache_path=args.cache)
    ranker = build_from_legacy(args.pretrained_checkpoint, device="cpu")
    if not isinstance(ranker, GMF):
        raise TypeError("continuous flow ranking requires the released GMF model")
    flow, flow_metadata = load_exposure_flow(
        args.flow_checkpoint, ranker.item_embedding.weight.detach(), device="cpu"
    )
    config = ContinuousRankerConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        discriminator_reg=args.discriminator_reg,
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        early_stopping_patience=args.early_stopping,
        minimum_ndcg_improvement=args.minimum_ndcg_improvement,
        candidates_per_user=args.candidates_per_user,
        pool_batch_size=args.pool_batch_size,
        pool_refresh_epochs=args.pool_refresh_epochs,
        flow_steps=args.flow_steps,
        flow_solver=args.flow_solver,
        negative_source=args.negative_source,
        freeze_item_embeddings=not args.train_item_embeddings,
        selection_mode=args.selection_mode,
        hardness_beta=args.hardness_beta,
        minimum_ess_ratio=args.minimum_ess_ratio,
        safety_temperature=args.safety_temperature,
        transport_neighbors=args.transport_neighbors,
        transport_minimum_ess_ratio=args.transport_minimum_ess_ratio,
        transport_max_inverse_temperature=args.transport_max_inverse_temperature,
        dns_uniform_candidates=args.dns_uniform_candidates,
        dns_flow_candidates=args.dns_flow_candidates,
        ranking_loss=args.ranking_loss,
        hard_bpr_a=args.hard_bpr_a,
        hard_bpr_b=args.hard_bpr_b,
        hard_bpr_c=args.hard_bpr_c,
        recommendation_list_length=args.recommendation_list_length,
        eval_batch_size=args.eval_batch_size,
        maximum_train_examples=args.maximum_train_examples,
        diagnostics_examples=args.diagnostics_examples,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
    )
    trainer = ContinuousFlowRankerTrainer(data, ranker, flow, config)
    print(
        json.dumps(
            {
                "device": str(trainer.device),
                "negative_source": config.negative_source,
                "item_embeddings_frozen": config.freeze_item_embeddings,
                "continuous_training_without_catalog_mapping": (
                    config.negative_source == "flow"
                ),
                "flow_epoch": flow_metadata.get("epoch"),
                "official_test_used": False,
            },
            indent=2,
        ),
        flush=True,
    )
    trainer.fit(args.output)


def command_audit(args: argparse.Namespace) -> None:
    data = InteractionData.load(args.data, cache_path=args.cache)
    device = resolve_device(args.device)
    catalog = _catalog(args.pretrained_checkpoint, device)
    flow, metadata = load_exposure_flow(
        args.flow_checkpoint, catalog, device=device
    )
    training = metadata.get("training_config")
    if not isinstance(training, dict):
        raise ValueError("flow checkpoint does not record its training configuration")
    _, holdout = split_train_exposures(
        data.train_exposures.keys,
        validation_fraction=float(training["validation_fraction"]),
        validation_examples=int(training["validation_examples"]),
        seed=int(training["seed"]) + 103,
    )
    config = ExposureFlowAuditConfig(
        examples=args.examples,
        nearest_examples=args.nearest_examples,
        projections=args.projections,
        solver_steps=tuple(args.solver_steps),
        c2st_epochs=args.c2st_epochs,
        c2st_hidden_dim=args.c2st_hidden_dim,
        seed=args.seed,
    )
    result = audit_exposure_flow(data, flow, catalog, holdout, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "pretrain":
        command_pretrain(args)
    elif args.command == "inspect":
        command_inspect(args)
    elif args.command == "rank-continuous":
        command_rank_continuous(args)
    else:
        command_audit(args)
