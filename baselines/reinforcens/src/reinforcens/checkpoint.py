from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import TrainConfig
from .models import BarycentricGenerator, GMF, MLP, Recommender, build_model


def _unwrapped(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def read_legacy_pickle(path: str | Path) -> list[Any]:
    """Read the released Python-2/cPickle parameter file safely on Python 3."""

    with Path(path).expanduser().open("rb") as stream:
        try:
            values = pickle.load(stream, encoding="latin1")
        except TypeError:
            stream.seek(0)
            values = pickle.load(stream)
    if not isinstance(values, list) or len(values) < 3:
        raise ValueError(f"Not a ReinforceNS parameter pickle: {path}")
    return values


def infer_legacy_architecture(values: list[Any]) -> tuple[str, int, int, int, int]:
    user, item, h = (np.asarray(values[index]) for index in range(3))
    if user.ndim != 2 or item.ndim != 2:
        raise ValueError("legacy user/item embeddings must be matrices")
    if user.shape[1] != item.shape[1]:
        raise ValueError("legacy embedding dimensions do not match")
    embedding_dim = int(user.shape[1])
    h_size = int(h.size)
    if h_size == embedding_dim and len(values) == 3:
        architecture, layer_count = "gmf", 0
    else:
        architecture, layer_count = "mlp", len(values) - 3
        expected = 2 * embedding_dim // (2**layer_count)
        if h_size != expected:
            raise ValueError(
                f"cannot infer legacy model: h has {h_size} values, expected {expected}"
            )
    return (
        architecture,
        int(user.shape[0]),
        int(item.shape[0]),
        embedding_dim,
        layer_count,
    )


def build_from_legacy(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> Recommender:
    values = read_legacy_pickle(path)
    architecture, num_users, num_items, embedding_dim, layer_count = (
        infer_legacy_architecture(values)
    )
    model = build_model(architecture, num_users, num_items, embedding_dim, layer_count)
    load_legacy_weights(model, values)
    return model.to(device)


def load_legacy_weights(model: Recommender, source: str | Path | list[Any]) -> None:
    values = read_legacy_pickle(source) if not isinstance(source, list) else source
    user, item, h = (np.asarray(values[index], dtype=np.float32) for index in range(3))
    expected = (
        (model.num_users, model.embedding_dim),
        (model.num_items, model.embedding_dim),
    )
    if user.shape != expected[0] or item.shape != expected[1]:
        raise ValueError(
            f"legacy shapes {(user.shape, item.shape)} do not match {expected}"
        )
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.from_numpy(user))  # type: ignore[attr-defined]
        model.item_embedding.weight.copy_(torch.from_numpy(item))  # type: ignore[attr-defined]
        model.h.copy_(torch.from_numpy(h.reshape(-1)))  # type: ignore[attr-defined]
        if isinstance(model, MLP):
            if len(values) != 3 + len(model.layers):
                raise ValueError("legacy MLP layer count does not match model")
            for layer, pair in zip(model.layers, values[3:]):
                weight, bias = (np.asarray(value, dtype=np.float32) for value in pair)
                # Released dense matrices are [input, output]; PyTorch uses [output, input].
                layer.weight.copy_(torch.from_numpy(weight.T))
                layer.bias.copy_(torch.from_numpy(bias.reshape(-1)))
        elif not isinstance(model, GMF):
            raise TypeError(f"legacy loading is unsupported for {type(model).__name__}")


def save_checkpoint(
    path: str | Path,
    *,
    config: TrainConfig,
    discriminator: Recommender,
    generator: nn.Module | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "config": config.to_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
            "history": history or [],
            "discriminator": _unwrapped(discriminator).state_dict(),
            "generator": None
            if generator is None
            else _unwrapped(generator).state_dict(),
        },
        target,
    )


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TrainConfig, Recommender, nn.Module | None, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    config = TrainConfig.from_dict(payload["config"])
    discriminator = build_model(
        config.architecture,
        config.num_users,
        config.num_items,
        config.embedding_dim,
        config.mlp_layers,
    ).to(device)
    discriminator.load_state_dict(payload["discriminator"])
    generator = None
    if payload.get("generator") is not None:
        policy = build_model(
            config.architecture,
            config.num_users,
            config.num_items,
            config.embedding_dim,
            config.mlp_layers,
        ).to(device)
        if config.model in {"berns", "cberns", "ehrns"}:
            exposure_prior = build_model(
                config.architecture,
                config.num_users,
                config.num_items,
                config.embedding_dim,
                config.mlp_layers,
            ).to(device)
            generator = BarycentricGenerator(policy, exposure_prior).to(device)
        else:
            generator = policy
        generator.load_state_dict(payload["generator"])
        if isinstance(generator, BarycentricGenerator):
            epoch = int(payload.get("epoch", 0))
            prior_weight = (
                1.0 / (epoch + 1)
                if config.model in {"cberns", "ehrns"}
                else 0.5
            )
            generator.set_prior_weight(prior_weight)
    metadata = {key: payload.get(key) for key in ("epoch", "metrics", "history")}
    return config, discriminator, generator, metadata


def describe_legacy(path: str | Path) -> dict[str, Any]:
    values = read_legacy_pickle(path)
    architecture, num_users, num_items, embedding_dim, layer_count = (
        infer_legacy_architecture(values)
    )
    return {
        "path": str(Path(path).resolve()),
        "architecture": architecture,
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": embedding_dim,
        "mlp_layers": layer_count,
        "parameter_arrays": [
            list(np.asarray(value).shape) if not isinstance(value, list) else "layer"
            for value in values
        ],
    }


def write_checkpoint_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
