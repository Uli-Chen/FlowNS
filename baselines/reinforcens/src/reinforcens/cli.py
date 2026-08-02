from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .checkpoint import (
    build_from_legacy,
    describe_legacy,
    load_checkpoint,
    save_checkpoint,
)
from .config import TrainConfig
from .data import InteractionData
from .metrics import evaluate_list, evaluate_topk, format_result
from .trainer import Trainer, resolve_device


BASELINE_ROOT = Path(__file__).resolve().parents[2]
# Keep the public name for compatibility with downstream EH-RNS/Flow-RNS tools.
REPOSITORY_ROOT = BASELINE_ROOT
DEFAULT_DATA = REPOSITORY_ROOT / "data" / "zhihu" / "train-valid-test.zip"
DEFAULT_PRETRAINED = (
    REPOSITORY_ROOT / "checkpoints" / "pretrain_model_dis_zhihu.pkl"
)


def _sigma_range(value: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(part) for part in value.strip("[]").split(",") if part.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values or any(sigma <= 0 for sigma in values):
        raise argparse.ArgumentTypeError("sigma range must contain positive numbers")
    return values


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Zhihu directory or train-valid-test.zip",
    )
    parser.add_argument(
        "--cache", type=Path, default=REPOSITORY_ROOT / ".cache" / "zhihu.npz"
    )
    parser.add_argument("--num-users", type=int, default=16_015)
    parser.add_argument("--num-items", type=int, default=45_782)
    parser.add_argument(
        "--keep-exposure-conflicts",
        action="store_true",
        help="Reproduce the legacy list-mode bug that keeps items clicked in another session as negatives",
    )


def _load_data(args: argparse.Namespace) -> InteractionData:
    print(f"Loading data from {args.data} ...", flush=True)
    data = InteractionData.load(
        args.data,
        num_users=args.num_users,
        num_items=args.num_items,
        clean_exposure_conflicts=not args.keep_exposure_conflicts,
        cache_path=args.cache,
    )
    print(json.dumps(data.statistics(), indent=2, ensure_ascii=False), flush=True)
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch ReinforceNS baselines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        help="Train BPR, DNS, KBGAN, RNS, EP-RNS, BE-RNS, CBE-RNS, EH-RNS, or ItemPop",
    )
    _add_data_arguments(train)
    train.add_argument(
        "--model",
        choices=(
            "bpr",
            "dns",
            "kbgan",
            "rns",
            "eprns",
            "berns",
            "cberns",
            "ehrns",
            "itempop",
        ),
        default="rns",
    )
    train.add_argument("--architecture", choices=("gmf", "mlp"), default="gmf")
    train.add_argument("--embedding-dim", type=int, default=32)
    train.add_argument("--mlp-layers", type=int, default=0)
    train.add_argument("--epochs", type=int, default=400)
    train.add_argument("--batch-size", type=int, default=1_024)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--generator-learning-rate", type=float)
    train.add_argument(
        "--optimizer", choices=("adam", "adagrad", "sgd"), default="adam"
    )
    train.add_argument("--regularization", type=float, default=1e-5)
    train.add_argument("--generator-regularization", type=float, default=1e-5)
    train.add_argument("--num-negatives", type=int, default=1)
    train.add_argument("--candidates", type=int, default=30)
    train.add_argument("--exposure-candidates", type=int, default=1)
    train.add_argument("--dns-candidates", type=int, default=50)
    train.add_argument("--temperature", type=float, default=1.0)
    train.add_argument("--alpha", type=float, default=2.5)
    train.add_argument("--beta", type=float, default=0.75)
    train.add_argument(
        "--sigma-range", type=_sigma_range, default=(19.0, 20.0, 21.0, 22.0, 23.0)
    )
    train.add_argument("--entropy-target", type=float, default=1.0)
    train.add_argument("--full-candidates", action="store_true")
    train.add_argument("--no-dns-loss", action="store_true")
    train.add_argument("--freeze-generator", action="store_true")
    train.add_argument("--exposure-pretrain-ratio", type=float, default=0.0)
    train.add_argument("--generator-pretrain-epochs", type=int, default=0)
    train.add_argument("--generator-pretrain-batch-size", type=int, default=16_384)
    train.add_argument("--generator-pretrain-negatives", type=int, default=16)
    train.add_argument("--generator-validation-fraction", type=float, default=0.05)
    train.add_argument("--generator-validation-examples", type=int, default=131_072)
    train.add_argument(
        "--pretrained-checkpoint",
        "--legacy-checkpoint",
        dest="pretrained_checkpoint",
        type=Path,
        help="Initialize the ranker and sampler from the released BPR-GMF cPickle",
    )
    train.add_argument("--eval-mode", choices=("list", "topk"), default="list")
    train.add_argument("--list-length", type=int, default=160)
    train.add_argument("--top-k", type=int, default=100)
    train.add_argument("--eval-batch-size", type=int, default=1_024)
    train.add_argument("--eval-every", type=int, default=1)
    train.add_argument("--early-stopping", type=int, default=10)
    train.add_argument("--select-by", choices=("auc", "hr", "ndcg"), default="auc")
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--device", default="auto")
    train.add_argument("--amp", action="store_true")
    train.add_argument("--compile", action="store_true")
    train.add_argument("--no-drop-last", action="store_true")
    train.add_argument("--max-train-examples", type=int)
    train.add_argument("--output", type=Path, default=BASELINE_ROOT / "runs" / "rns")

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate the released BPR-GMF or a trained PyTorch checkpoint"
    )
    _add_data_arguments(evaluate)
    checkpoint_group = evaluate.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--pretrained-checkpoint",
        "--legacy-checkpoint",
        dest="pretrained_checkpoint",
        type=Path,
    )
    checkpoint_group.add_argument("--checkpoint", type=Path)
    evaluate.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    evaluate.add_argument("--eval-mode", choices=("list", "topk"), default="list")
    evaluate.add_argument("--list-length", type=int, default=160)
    evaluate.add_argument("--top-k", type=int, default=100)
    evaluate.add_argument("--batch-size", type=int, default=1_024)
    evaluate.add_argument("--seed", type=int, default=1)
    evaluate.add_argument("--device", default="auto")

    inspect_data = subparsers.add_parser(
        "inspect-data", help="Parse data and print statistics"
    )
    _add_data_arguments(inspect_data)
    inspect_data.add_argument("--build-eval-lists", action="store_true")
    inspect_data.add_argument("--list-length", type=int, default=160)

    convert = subparsers.add_parser(
        "convert", help="Convert the released BPR-GMF cPickle to .pt"
    )
    convert.add_argument("--input", type=Path, default=DEFAULT_PRETRAINED)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--model", choices=("bpr", "kbgan", "rns"), default="bpr")
    return parser


def _train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        model=args.model,
        architecture=args.architecture,
        num_users=args.num_users,
        num_items=args.num_items,
        embedding_dim=args.embedding_dim,
        mlp_layers=args.mlp_layers,
        learning_rate=args.learning_rate,
        generator_learning_rate=args.generator_learning_rate,
        optimizer=args.optimizer,
        discriminator_reg=args.regularization,
        generator_reg=args.generator_regularization,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_negatives=args.num_negatives,
        candidates=args.candidates,
        exposure_candidates=args.exposure_candidates,
        dns_candidates=args.dns_candidates,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        sigma_range=args.sigma_range,
        entropy_target=args.entropy_target,
        reduced=not args.full_candidates,
        no_dns_loss=args.no_dns_loss,
        freeze_generator=args.freeze_generator,
        exposure_pretrain_ratio=args.exposure_pretrain_ratio,
        generator_pretrain_epochs=args.generator_pretrain_epochs,
        generator_pretrain_batch_size=args.generator_pretrain_batch_size,
        generator_pretrain_negatives=args.generator_pretrain_negatives,
        generator_validation_fraction=args.generator_validation_fraction,
        generator_validation_examples=args.generator_validation_examples,
        eval_mode=args.eval_mode,
        recommendation_list_length=args.list_length,
        top_k=args.top_k,
        eval_batch_size=args.eval_batch_size,
        eval_every=args.eval_every,
        early_stopping_patience=args.early_stopping,
        select_by=args.select_by,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        compile_model=args.compile,
        drop_last=not args.no_drop_last,
        clean_exposure_conflicts=not args.keep_exposure_conflicts,
        max_train_examples=args.max_train_examples,
    )


def command_train(args: argparse.Namespace) -> None:
    config = _train_config(args)
    if (
        config.model in {"kbgan", "rns", "eprns", "berns", "cberns", "ehrns"}
        and config.num_negatives != 1
    ):
        raise SystemExit(
            "KBGAN/ReinforceNS support exactly one generated ranker negative"
        )
    data = _load_data(args)
    trainer = Trainer(
        data, config, pretrained_checkpoint=args.pretrained_checkpoint
    )
    print(
        f"Training {config.model.upper()} on {trainer.device}; output={args.output}",
        flush=True,
    )
    trainer.fit(args.output)


def command_evaluate(args: argparse.Namespace) -> None:
    data = _load_data(args)
    device = resolve_device(args.device)
    if args.pretrained_checkpoint:
        model = build_from_legacy(args.pretrained_checkpoint, device=device)
    else:
        _, model, _, _ = load_checkpoint(args.checkpoint, device=device)
    if model.num_users != data.num_users or model.num_items != data.num_items:
        raise SystemExit("checkpoint dimensions do not match the dataset")
    if args.eval_mode == "list":
        candidates = data.make_eval_candidates(args.split, args.list_length, args.seed)
        result = evaluate_list(
            model, candidates, device=device, batch_size=args.batch_size
        )
    else:
        result = evaluate_topk(
            model,
            data,
            split=args.split,
            k=args.top_k,
            device=device,
            batch_size=min(args.batch_size, 256),
        )
    print(format_result(result))
    print(json.dumps(result.to_dict(), indent=2))


def command_inspect_data(args: argparse.Namespace) -> None:
    data = _load_data(args)
    if args.build_eval_lists:
        for split in ("validation", "test"):
            candidates = data.make_eval_candidates(split, args.list_length)
            print(
                f"{split}: shape={candidates.items.shape}, positives={int(candidates.labels.sum())}, "
                f"negatives={int((~candidates.labels).sum())}"
            )


def command_convert(args: argparse.Namespace) -> None:
    description = describe_legacy(args.input)
    model = build_from_legacy(args.input)
    config = TrainConfig(
        model=args.model,
        architecture=description["architecture"],
        num_users=description["num_users"],
        num_items=description["num_items"],
        embedding_dim=description["embedding_dim"],
        mlp_layers=description["mlp_layers"],
        epochs=0,
    )
    generator = model if args.model in {"kbgan", "rns"} else None
    save_checkpoint(
        args.output,
        config=config,
        discriminator=model,
        generator=generator,
        metrics={"source": str(args.input.resolve())},  # type: ignore[dict-item]
    )
    print(json.dumps(description, indent=2))
    print(f"Converted checkpoint written to {args.output}")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    commands = {
        "train": command_train,
        "evaluate": command_evaluate,
        "inspect-data": command_inspect_data,
        "convert": command_convert,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
