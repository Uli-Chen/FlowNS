from __future__ import annotations

import contextlib
import io
import json
import random
import zipfile
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np


SPLIT_FILENAMES = {
    "train": "data.zhihu.display.train",
    "validation": "data.zhihu.display.validation",
    "test": "data.zhihu.display.test",
}


@dataclass(slots=True)
class CSRItems:
    """Compact per-user item sets plus vectorized membership tests."""

    num_users: int
    num_items: int
    indptr: np.ndarray
    items: np.ndarray
    keys: np.ndarray

    @classmethod
    def from_keys(cls, keys: np.ndarray, num_users: int, num_items: int) -> "CSRItems":
        keys = np.unique(np.asarray(keys, dtype=np.int64))
        if keys.size:
            users = keys // num_items
            if users[-1] >= num_users:
                raise ValueError(
                    f"Encountered user {users[-1]}, but num_users={num_users}"
                )
            counts = np.bincount(users, minlength=num_users)
            items = (keys % num_items).astype(np.int32, copy=False)
        else:
            counts = np.zeros(num_users, dtype=np.int64)
            items = np.empty(0, dtype=np.int32)
        indptr = np.empty(num_users + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        return cls(num_users, num_items, indptr, items, keys)

    @classmethod
    def empty(cls, num_users: int, num_items: int) -> "CSRItems":
        return cls.from_keys(np.empty(0, dtype=np.int64), num_users, num_items)

    def for_user(self, user: int) -> np.ndarray:
        return self.items[self.indptr[user] : self.indptr[user + 1]]

    def counts(self) -> np.ndarray:
        return np.diff(self.indptr)

    def users_with_items(self) -> np.ndarray:
        return np.flatnonzero(np.diff(self.indptr)).astype(np.int32)

    def contains(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        keys = users * self.num_items + items
        flat = keys.reshape(-1)
        positions = np.searchsorted(self.keys, flat)
        result = np.zeros(flat.size, dtype=bool)
        valid = positions < self.keys.size
        result[valid] = self.keys[positions[valid]] == flat[valid]
        return result.reshape(keys.shape)

    def without(self, other: "CSRItems") -> "CSRItems":
        keys = np.setdiff1d(self.keys, other.keys, assume_unique=True)
        return CSRItems.from_keys(keys, self.num_users, self.num_items)


@dataclass(slots=True)
class InteractionSplit:
    clicks: CSRItems
    exposures: CSRItems


@dataclass(slots=True)
class EvalCandidates:
    users: np.ndarray
    items: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        if self.items.shape != self.labels.shape:
            raise ValueError("items and labels must have identical shapes")
        if self.items.shape[0] != self.users.shape[0]:
            raise ValueError("one candidate row is required per user")


@dataclass(slots=True)
class InteractionData:
    num_users: int
    num_items: int
    train_users: np.ndarray
    train_items: np.ndarray
    train_clicks: CSRItems
    train_exposures: CSRItems
    validation: InteractionSplit
    test: InteractionSplit
    source: str
    exposure_conflicts_removed: int = 0

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        num_users: int = 16_015,
        num_items: int = 45_782,
        clean_exposure_conflicts: bool = True,
        cache_path: str | Path | None = None,
    ) -> "InteractionData":
        path = Path(path).expanduser().resolve()
        cache = Path(cache_path).expanduser().resolve() if cache_path else None
        fingerprint = _source_fingerprint(
            path, num_users, num_items, clean_exposure_conflicts
        )
        if cache is not None and cache.exists():
            cached = cls._load_cache(cache)
            if cached[0] == fingerprint:
                return cached[1]

        train_users_buf = array("i")
        train_items_buf = array("i")
        train_click_keys = array("q")
        train_exposure_keys = array("q")
        with _open_split(path, "train") as stream:
            _parse_stream(
                stream,
                num_items,
                train_click_keys,
                train_exposure_keys,
                train_users_buf,
                train_items_buf,
            )

        train_users = np.frombuffer(train_users_buf, dtype=np.int32).copy()
        train_items = np.frombuffer(train_items_buf, dtype=np.int32).copy()
        train_clicks = CSRItems.from_keys(
            np.frombuffer(train_click_keys, dtype=np.int64), num_users, num_items
        )
        train_exposures = CSRItems.from_keys(
            np.frombuffer(train_exposure_keys, dtype=np.int64), num_users, num_items
        )
        conflicts = int(
            np.intersect1d(
                train_clicks.keys, train_exposures.keys, assume_unique=True
            ).size
        )
        if clean_exposure_conflicts:
            train_exposures = train_exposures.without(train_clicks)

        validation = _load_eval_split(path, "validation", num_users, num_items)
        test = _load_eval_split(path, "test", num_users, num_items)
        result = cls(
            num_users=num_users,
            num_items=num_items,
            train_users=train_users,
            train_items=train_items,
            train_clicks=train_clicks,
            train_exposures=train_exposures,
            validation=validation,
            test=test,
            source=str(path),
            exposure_conflicts_removed=conflicts if clean_exposure_conflicts else 0,
        )
        result.validate()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            result._save_cache(cache, fingerprint)
        return result

    def validate(self) -> None:
        if self.train_users.shape != self.train_items.shape:
            raise ValueError("train user and item arrays have different sizes")
        if self.train_users.size == 0:
            raise ValueError("training data contains no clicks")
        if self.train_users.min() < 0 or self.train_users.max() >= self.num_users:
            raise ValueError("training user id outside configured range")
        if self.train_items.min() < 0 or self.train_items.max() >= self.num_items:
            raise ValueError("training item id outside configured range")

    def statistics(self) -> dict[str, int | str]:
        return {
            "source": self.source,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "train_click_events": int(self.train_items.size),
            "train_unique_clicks": int(self.train_clicks.items.size),
            "train_unique_exposures": int(self.train_exposures.items.size),
            "exposure_conflicts_removed": self.exposure_conflicts_removed,
            "validation_users": int(self.validation.clicks.users_with_items().size),
            "test_users": int(self.test.clicks.users_with_items().size),
        }

    def make_eval_candidates(
        self,
        split: str,
        length: int = 160,
        seed: int = 1,
    ) -> EvalCandidates:
        """Build the paper's mixed click/display recommendation lists."""

        if length < 2:
            raise ValueError("recommendation-list evaluation requires length >= 2")
        if split not in {"validation", "test"}:
            raise ValueError("split must be validation or test")
        rng = random.Random(seed)
        # The original Dataset constructor builds validation first and test second
        # under one random.seed(1) stream. Replay validation draws when test is
        # requested independently so CLI evaluation has the same protocol.
        if split == "test":
            self._make_eval_candidates_with_rng("validation", length, rng)
        return self._make_eval_candidates_with_rng(split, length, rng)

    def _make_eval_candidates_with_rng(
        self,
        split: str,
        length: int,
        rng: random.Random,
    ) -> EvalCandidates:
        split_data = self.validation if split == "validation" else self.test
        users = split_data.clicks.users_with_items()
        items_out = np.empty((users.size, length), dtype=np.int32)
        labels_out = np.zeros((users.size, length), dtype=np.bool_)

        for row, user_np in enumerate(users):
            user = int(user_np)
            clicks = split_data.clicks.for_user(user).tolist()
            click_set = set(clicks)
            exposures = [
                int(i)
                for i in split_data.exposures.for_user(user)
                if int(i) not in click_set
            ]
            forbidden = set(int(i) for i in self.train_clicks.for_user(user))
            if split == "test":
                forbidden.update(int(i) for i in self.validation.clicks.for_user(user))
            forbidden.update(click_set)
            forbidden.update(exposures)

            total = len(clicks) + len(exposures)
            fillers: list[int] = []
            if total <= length:
                while len(fillers) < length - total:
                    candidate = rng.randrange(self.num_items)
                    if candidate not in forbidden:
                        forbidden.add(candidate)
                        fillers.append(candidate)
                selected_clicks = clicks
                selected_exposures = exposures
            else:
                if not exposures:
                    candidate = rng.randrange(self.num_items)
                    while candidate in forbidden:
                        candidate = rng.randrange(self.num_items)
                    exposures = [candidate]
                total = len(clicks) + len(exposures)
                exposure_count = max(1, int(len(exposures) / float(total) * length))
                click_count = length - exposure_count
                if click_count > len(clicks):
                    click_count = len(clicks)
                    exposure_count = length - click_count
                if exposure_count > len(exposures):
                    exposure_count = len(exposures)
                    click_count = length - exposure_count
                selected_exposures = rng.sample(exposures, exposure_count)
                selected_clicks = rng.sample(clicks, click_count)
                while (
                    len(selected_exposures) + len(selected_clicks) + len(fillers)
                    < length
                ):
                    candidate = rng.randrange(self.num_items)
                    if candidate not in forbidden:
                        forbidden.add(candidate)
                        fillers.append(candidate)

            candidates = fillers + selected_exposures + selected_clicks
            if len(candidates) != length or not selected_clicks:
                raise RuntimeError(
                    f"failed to construct evaluation row for user {user}"
                )
            items_out[row] = candidates
            labels_out[row, -len(selected_clicks) :] = True
        return EvalCandidates(users=users, items=items_out, labels=labels_out)

    def _save_cache(self, path: Path, fingerprint: str) -> None:
        np.savez(
            path,
            metadata=np.array(
                json.dumps({"fingerprint": fingerprint, "source": self.source})
            ),
            num_users=np.array(self.num_users),
            num_items=np.array(self.num_items),
            train_users=self.train_users,
            train_items=self.train_items,
            train_click_keys=self.train_clicks.keys,
            train_exposure_keys=self.train_exposures.keys,
            validation_click_keys=self.validation.clicks.keys,
            validation_exposure_keys=self.validation.exposures.keys,
            test_click_keys=self.test.clicks.keys,
            test_exposure_keys=self.test.exposures.keys,
            conflicts=np.array(self.exposure_conflicts_removed),
        )

    @classmethod
    def _load_cache(cls, path: Path) -> tuple[str, "InteractionData"]:
        with np.load(path, allow_pickle=False) as values:
            metadata = json.loads(str(values["metadata"]))
            num_users = int(values["num_users"])
            num_items = int(values["num_items"])

            def make_csr(name: str) -> CSRItems:
                return CSRItems.from_keys(values[name], num_users, num_items)

            result = cls(
                num_users=num_users,
                num_items=num_items,
                train_users=values["train_users"].astype(np.int32, copy=True),
                train_items=values["train_items"].astype(np.int32, copy=True),
                train_clicks=make_csr("train_click_keys"),
                train_exposures=make_csr("train_exposure_keys"),
                validation=InteractionSplit(
                    make_csr("validation_click_keys"),
                    make_csr("validation_exposure_keys"),
                ),
                test=InteractionSplit(
                    make_csr("test_click_keys"), make_csr("test_exposure_keys")
                ),
                source=metadata["source"],
                exposure_conflicts_removed=int(values["conflicts"]),
            )
        result.validate()
        return metadata["fingerprint"], result


def _load_eval_split(
    path: Path, split: str, num_users: int, num_items: int
) -> InteractionSplit:
    clicks = array("q")
    exposures = array("q")
    with _open_split(path, split) as stream:
        _parse_stream(stream, num_items, clicks, exposures)
    return InteractionSplit(
        CSRItems.from_keys(np.frombuffer(clicks, dtype=np.int64), num_users, num_items),
        CSRItems.from_keys(
            np.frombuffer(exposures, dtype=np.int64), num_users, num_items
        ),
    )


def _parse_stream(
    stream: TextIO,
    num_items: int,
    click_keys: array,
    exposure_keys: array,
    train_users: array | None = None,
    train_items: array | None = None,
) -> None:
    for line_number, line in enumerate(stream, 1):
        fields = line.rstrip("\n").split(",", 4)
        if len(fields) != 5:
            raise ValueError(
                f"Malformed interaction at line {line_number}: {line[:80]!r}"
            )
        user = int(fields[0])
        event = fields[3]
        if event == "event_click":
            item = int(fields[1])
            click_keys.append(user * num_items + item)
            if train_users is not None and train_items is not None:
                train_users.append(user)
                train_items.append(item)
        elif event == "list_show":
            prefix = user * num_items
            exposure_keys.extend(
                prefix + int(item) for item in fields[1].split("|") if item
            )


@contextlib.contextmanager
def _open_split(path: Path, split: str) -> Iterator[TextIO]:
    filename = SPLIT_FILENAMES[split]
    if path.is_dir():
        with (path / filename).open("r", encoding="utf-8") as stream:
            yield stream
        return
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            with archive.open(filename, "r") as raw:
                with io.TextIOWrapper(raw, encoding="utf-8") as stream:
                    yield stream
        return
    raise ValueError(f"Data path must be the dataset directory or zip archive: {path}")


def _source_fingerprint(path: Path, num_users: int, num_items: int, clean: bool) -> str:
    stat = path.stat()
    return json.dumps(
        {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "num_users": num_users,
            "num_items": num_items,
            "clean": clean,
            "cache_version": 2,
        },
        sort_keys=True,
    )
