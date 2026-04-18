from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from .config import load_config
from .trainer import train_from_config


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="flowns")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a minimal recommender")
    train_parser.add_argument("--config", required=True, help="Path to a YAML config file")
    train_parser.add_argument("--model", choices=["mf", "sasrec"], help="Override model in config")
    train_parser.add_argument("--wandb", choices=["disabled", "offline", "online"], help="Override wandb mode")

    args = parser.parse_args(argv)
    if args.command == "train":
        config = _apply_overrides(load_config(args.config), args)
        print(json.dumps(train_from_config(config), indent=2, sort_keys=True))


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.model:
        config["model"] = args.model
    if args.wandb:
        wandb_config = config.setdefault("wandb", {})
        wandb_config["enabled"] = args.wandb != "disabled"
        if args.wandb != "disabled":
            wandb_config["mode"] = args.wandb
    return config
