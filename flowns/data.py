from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union


@dataclass(frozen=True)
class Interaction:
    user: int
    item: int
    timestamp: float
    rating: float


@dataclass(frozen=True)
class SequenceSample:
    user: int
    sequence: tuple[int, ...]
    target: int


@dataclass
class RecommenderData:
    num_users: int
    num_items: int
    user_tokens: list[str]
    item_tokens: list[str]
    train_pairs: list[tuple[int, int]]
    train_by_user: dict[int, set[int]]
    positives_by_user: dict[int, set[int]]
    valid_targets: dict[int, int]
    test_targets: dict[int, int]
    seq_train_samples: list[SequenceSample]
    seq_valid_samples: list[SequenceSample]
    seq_test_samples: list[SequenceSample]


def load_recommender_data(config: dict[str, Any]) -> RecommenderData:
    interactions = load_interactions(_resolve_inter_file(config), config.get("min_rating"))
    return build_recommender_data(interactions)


def _resolve_inter_file(config: dict[str, Any]) -> Path:
    data_config = config.get("data", {})
    if "inter_file" in data_config:
        return Path(data_config["inter_file"])

    data_path = Path(data_config.get("data_path", "dataset"))
    dataset = data_config.get("dataset")
    if not dataset:
        raise ValueError("Missing config data.dataset or data.inter_file")
    return data_path / dataset / f"{dataset}.inter"


def load_interactions(path: Union[str, Path], min_rating: Optional[float] = None) -> list[Interaction]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        fields = [column.split(":", 1)[0] for column in header]
        try:
            user_idx = fields.index("user_id")
            item_idx = fields.index("item_id")
        except ValueError as exc:
            raise ValueError(f"{path} must contain user_id and item_id columns") from exc

        rating_idx = fields.index("rating") if "rating" in fields else None
        timestamp_idx = fields.index("timestamp") if "timestamp" in fields else None

        rows: list[tuple[str, str, float, float]] = []
        for line_no, raw_line in enumerate(handle, start=2):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            parts = raw_line.split()
            if len(parts) < len(fields):
                raise ValueError(f"Malformed row in {path}:{line_no}")
            rating = float(parts[rating_idx]) if rating_idx is not None else 1.0
            if min_rating is not None and rating < float(min_rating):
                continue
            timestamp = float(parts[timestamp_idx]) if timestamp_idx is not None else float(line_no)
            rows.append((parts[user_idx], parts[item_idx], timestamp, rating))

    user_map, user_tokens = _build_token_map(row[0] for row in rows)
    item_map, item_tokens = _build_token_map(row[1] for row in rows)
    return [
        Interaction(user_map[user], item_map[item], timestamp, rating)
        for user, item, timestamp, rating in rows
    ]


def build_recommender_data(interactions: list[Interaction]) -> RecommenderData:
    if not interactions:
        raise ValueError("No interactions available after filtering")

    num_users = max(inter.user for inter in interactions) + 1
    num_items = max(inter.item for inter in interactions) + 1
    by_user: dict[int, list[Interaction]] = {user: [] for user in range(num_users)}
    positives_by_user: dict[int, set[int]] = {user: set() for user in range(num_users)}

    for inter in interactions:
        by_user[inter.user].append(Interaction(inter.user, inter.item, inter.timestamp, inter.rating))
        positives_by_user[inter.user].add(inter.item)

    train_pairs: list[tuple[int, int]] = []
    train_by_user: dict[int, set[int]] = {user: set() for user in range(num_users)}
    valid_targets: dict[int, int] = {}
    test_targets: dict[int, int] = {}
    seq_train_samples: list[SequenceSample] = []
    seq_valid_samples: list[SequenceSample] = []
    seq_test_samples: list[SequenceSample] = []

    for user, user_interactions in by_user.items():
        history = [inter.item for inter in sorted(user_interactions, key=lambda inter: inter.timestamp)]
        if not history:
            continue

        train_history, valid_item, test_item = _leave_one_out(history)
        for item in train_history:
            train_pairs.append((user, item))
            train_by_user[user].add(item)

        if valid_item is not None:
            valid_targets[user] = valid_item
            if train_history:
                seq_valid_samples.append(SequenceSample(user, tuple(train_history), valid_item))
        if test_item is not None:
            test_targets[user] = test_item
            test_sequence = tuple(train_history + ([valid_item] if valid_item is not None else []))
            if test_sequence:
                seq_test_samples.append(SequenceSample(user, test_sequence, test_item))

        for index in range(1, len(train_history)):
            seq_train_samples.append(
                SequenceSample(user, tuple(train_history[:index]), train_history[index])
            )

    user_tokens = [str(index) for index in range(num_users)]
    item_tokens = [str(index) for index in range(num_items)]
    return RecommenderData(
        num_users=num_users,
        num_items=num_items,
        user_tokens=user_tokens,
        item_tokens=item_tokens,
        train_pairs=train_pairs,
        train_by_user=train_by_user,
        positives_by_user=positives_by_user,
        valid_targets=valid_targets,
        test_targets=test_targets,
        seq_train_samples=seq_train_samples,
        seq_valid_samples=seq_valid_samples,
        seq_test_samples=seq_test_samples,
    )


def _leave_one_out(history: list[int]) -> tuple[list[int], Optional[int], Optional[int]]:
    if len(history) == 1:
        return history, None, None
    if len(history) == 2:
        return history[:1], None, history[-1]
    return history[:-2], history[-2], history[-1]


def _build_token_map(tokens: Iterable[str]) -> tuple[dict[str, int], list[str]]:
    token_map: dict[str, int] = {}
    ordered_tokens: list[str] = []
    for token in tokens:
        if token not in token_map:
            token_map[token] = len(token_map)
            ordered_tokens.append(token)
    return token_map, ordered_tokens
