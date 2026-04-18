from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import RecommenderData, SequenceSample, load_recommender_data
from .metrics import topk_metrics
from .models import MatrixFactorization, SASRec


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get("seed", 42)))
    data = load_recommender_data(config)
    model_name = str(config.get("model", "mf")).lower()

    if model_name in {"mf", "bpr", "matrix_factorization"}:
        result = train_general(config, data)
    elif model_name in {"sasrec", "sequential"}:
        result = train_sequential(config, data)
    else:
        raise ValueError(f"Unsupported model: {config.get('model')}")

    output_path = config.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def train_general(config: dict[str, Any], data: RecommenderData) -> dict[str, Any]:
    train_config = config.get("training", {})
    topks = config.get("evaluation", {}).get("topk", [10])
    device = _resolve_device(config)
    model = MatrixFactorization(
        data.num_users,
        data.num_items,
        embedding_dim=int(config.get("model_config", {}).get("embedding_dim", 64)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    run = _init_wandb(config, {"num_users": data.num_users, "num_items": data.num_items})

    best_valid = -1.0
    best_test: dict[str, float] = {}
    batch_size = int(train_config.get("batch_size", 1024))
    epochs = int(train_config.get("epochs", 20))

    for epoch in range(1, epochs + 1):
        random.shuffle(data.train_pairs)
        model.train()
        losses: list[float] = []
        for offset in tqdm(range(0, len(data.train_pairs), batch_size), disable=not config.get("show_progress", True)):
            batch = data.train_pairs[offset : offset + batch_size]
            users = torch.tensor([user for user, _ in batch], dtype=torch.long, device=device)
            positives = torch.tensor([item for _, item in batch], dtype=torch.long, device=device)
            negatives = torch.tensor(
                [_sample_negative(data.num_items, data.positives_by_user[user]) for user, _ in batch],
                dtype=torch.long,
                device=device,
            )
            loss = model.bpr_loss(users, positives, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        valid = evaluate_general(model, data, data.valid_targets, topks, device)
        test = evaluate_general(model, data, data.test_targets, topks, device)
        best_valid, best_test = _track_best(valid, test, topks, best_valid, best_test)
        _wandb_log(run, {"epoch": epoch, "loss": _mean(losses), **_prefix("valid", valid), **_prefix("test", test)})

    _wandb_finish(run)
    return {"model": "mf", "best_test": best_test, "num_users": data.num_users, "num_items": data.num_items}


def train_sequential(config: dict[str, Any], data: RecommenderData) -> dict[str, Any]:
    train_config = config.get("training", {})
    model_config = config.get("model_config", {})
    topks = config.get("evaluation", {}).get("topk", [10])
    device = _resolve_device(config)
    max_seq_len = int(model_config.get("max_seq_len", 50))
    model = SASRec(
        data.num_items,
        max_seq_len=max_seq_len,
        embedding_dim=int(model_config.get("embedding_dim", 64)),
        num_heads=int(model_config.get("num_heads", 2)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.2)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    run = _init_wandb(config, {"num_users": data.num_users, "num_items": data.num_items})

    loader = DataLoader(
        data.seq_train_samples,
        batch_size=int(train_config.get("batch_size", 256)),
        shuffle=True,
        collate_fn=lambda batch: _collate_sequence_batch(batch, data, max_seq_len, device),
    )
    epochs = int(train_config.get("epochs", 20))
    best_valid = -1.0
    best_test: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for sequences, positives, negatives in tqdm(loader, disable=not config.get("show_progress", True)):
            loss = model.bpr_loss(sequences, positives, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        valid = evaluate_sequential(model, data, data.seq_valid_samples, topks, max_seq_len, device)
        test = evaluate_sequential(model, data, data.seq_test_samples, topks, max_seq_len, device)
        best_valid, best_test = _track_best(valid, test, topks, best_valid, best_test)
        _wandb_log(run, {"epoch": epoch, "loss": _mean(losses), **_prefix("valid", valid), **_prefix("test", test)})

    _wandb_finish(run)
    return {"model": "sasrec", "best_test": best_test, "num_users": data.num_users, "num_items": data.num_items}


@torch.no_grad()
def evaluate_general(
    model: MatrixFactorization,
    data: RecommenderData,
    targets: dict[int, int],
    topks: list[int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    ranked: dict[int, list[int]] = {}
    k = min(max(topks), data.num_items)
    for user, target in targets.items():
        users = torch.tensor([user], dtype=torch.long, device=device)
        scores = model.score_all(users).squeeze(0)
        seen = set(data.train_by_user.get(user, set()))
        if targets is data.test_targets and user in data.valid_targets:
            seen.add(data.valid_targets[user])
        _mask_seen_items(scores, seen - {target})
        ranked[user] = torch.topk(scores, k=k).indices.detach().cpu().tolist()
    return topk_metrics(ranked, targets, topks)


@torch.no_grad()
def evaluate_sequential(
    model: SASRec,
    data: RecommenderData,
    samples: list[SequenceSample],
    topks: list[int],
    max_seq_len: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    ranked: dict[int, list[int]] = {}
    targets: dict[int, int] = {}
    k = min(max(topks), data.num_items)
    for sample in samples:
        sequence = _sequence_tensor(sample.sequence, max_seq_len, device)
        scores = model.score_all(sequence).squeeze(0)
        seen = set(sample.sequence) - {sample.target}
        _mask_seen_items(scores, seen)
        ranked[sample.user] = torch.topk(scores, k=k).indices.detach().cpu().tolist()
        targets[sample.user] = sample.target
    return topk_metrics(ranked, targets, topks)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _sample_negative(num_items: int, positives: set[int]) -> int:
    if len(positives) >= num_items:
        raise ValueError("Cannot sample negatives when user interacted with every item")
    while True:
        item = random.randrange(num_items)
        if item not in positives:
            return item


def _collate_sequence_batch(
    batch: list[SequenceSample],
    data: RecommenderData,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = torch.cat([_sequence_tensor(sample.sequence, max_seq_len, device) for sample in batch], dim=0)
    positives = torch.tensor([sample.target + 1 for sample in batch], dtype=torch.long, device=device)
    negatives = torch.tensor(
        [_sample_negative(data.num_items, data.positives_by_user[sample.user]) + 1 for sample in batch],
        dtype=torch.long,
        device=device,
    )
    return sequences, positives, negatives


def _sequence_tensor(sequence: tuple[int, ...], max_seq_len: int, device: torch.device) -> torch.Tensor:
    clipped = list(sequence[-max_seq_len:])
    padded = [0] * (max_seq_len - len(clipped)) + [item + 1 for item in clipped]
    return torch.tensor([padded], dtype=torch.long, device=device)


def _mask_seen_items(scores: torch.Tensor, seen_items: set[int]) -> None:
    if seen_items:
        scores[torch.tensor(list(seen_items), dtype=torch.long, device=scores.device)] = -torch.inf


def _track_best(
    valid: dict[str, float],
    test: dict[str, float],
    topks: list[int],
    best_valid: float,
    best_test: dict[str, float],
) -> tuple[float, dict[str, float]]:
    key = f"recall@{min(topks)}"
    current = valid.get(key, 0.0)
    if current >= best_valid:
        return current, test
    return best_valid, best_test


def _init_wandb(config: dict[str, Any], metadata: dict[str, Any]):
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None
    import wandb

    payload = {**config, **metadata}
    return wandb.init(
        project=wandb_config.get("project", "flowns"),
        name=wandb_config.get("run_name"),
        mode=wandb_config.get("mode", "offline"),
        config=payload,
    )


def _wandb_log(run, payload: dict[str, float]) -> None:
    if run is not None:
        run.log(payload)


def _wandb_finish(run) -> None:
    if run is not None:
        run.finish()


def _prefix(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in values.items()}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
