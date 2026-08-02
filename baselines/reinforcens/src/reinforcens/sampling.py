from __future__ import annotations

import numpy as np

from .data import CSRItems, InteractionData


class NegativeSampler:
    """Negative samplers corresponding to BPR, DNS, KBGAN, and RNS."""

    def __init__(self, data: InteractionData, seed: int = 1) -> None:
        self.data = data
        self.rng = np.random.default_rng(seed)

    def uniform(
        self,
        users: np.ndarray,
        count: int = 1,
        *,
        exclude_exposures: bool = False,
    ) -> np.ndarray:
        users = np.asarray(users, dtype=np.int32).reshape(-1)
        samples = self.rng.integers(
            0,
            self.data.num_items,
            size=(users.size, count),
            dtype=np.int32,
        )
        self._repair_collisions(
            users,
            samples,
            self.data.train_clicks,
            self.data.train_exposures if exclude_exposures else None,
        )
        return samples

    def exposed(self, users: np.ndarray, count: int = 1) -> np.ndarray:
        users = np.asarray(users, dtype=np.int32).reshape(-1)
        result = np.empty((users.size, count), dtype=np.int32)
        missing_rows: list[int] = []
        for row, user_np in enumerate(users):
            values = self.data.train_exposures.for_user(int(user_np))
            if values.size:
                result[row] = self.rng.choice(values, size=count, replace=True)
            else:
                missing_rows.append(row)
        if missing_rows:
            missing = np.asarray(missing_rows, dtype=np.int64)
            result[missing] = self.uniform(users[missing], count)
        return result

    def bpr(
        self, users: np.ndarray, count: int, exposure_ratio: float = 0.0
    ) -> np.ndarray:
        if exposure_ratio <= 0.0:
            return self.uniform(users, count)
        users = np.asarray(users, dtype=np.int32).reshape(-1)
        choose_exposed = self.rng.random((users.size, count)) < exposure_ratio
        result = self.uniform(users, count, exclude_exposures=True)
        exposed = self.exposed(users, count)
        return np.where(choose_exposed, exposed, result).astype(np.int32, copy=False)

    def kbgan_candidates(self, users: np.ndarray, count: int) -> np.ndarray:
        return self.uniform(users, count)

    def rns_candidates(
        self, users: np.ndarray, count: int, exposure_count: int
    ) -> np.ndarray:
        if exposure_count < 0 or exposure_count > count:
            raise ValueError("exposure_count must be between zero and count")
        unobserved_count = count - exposure_count
        parts: list[np.ndarray] = []
        if unobserved_count:
            parts.append(self.uniform(users, unobserved_count, exclude_exposures=True))
        if exposure_count:
            parts.append(self.exposed(users, exposure_count))
        if not parts:
            return np.empty((len(users), 0), dtype=np.int32)
        return np.concatenate(parts, axis=1)

    def is_exposed(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return self.data.train_exposures.contains(users, items)

    def _repair_collisions(
        self,
        users: np.ndarray,
        samples: np.ndarray,
        primary: CSRItems,
        secondary: CSRItems | None = None,
    ) -> None:
        expanded_users = np.broadcast_to(users[:, None], samples.shape)
        collisions = primary.contains(expanded_users, samples)
        if secondary is not None:
            collisions |= secondary.contains(expanded_users, samples)
        attempts = 0
        while collisions.any():
            samples[collisions] = self.rng.integers(
                0,
                self.data.num_items,
                size=int(collisions.sum()),
                dtype=np.int32,
            )
            collisions = primary.contains(expanded_users, samples)
            if secondary is not None:
                collisions |= secondary.contains(expanded_users, samples)
            attempts += 1
            if attempts > 100:
                raise RuntimeError(
                    "negative sampling did not converge; a user may cover the full item space"
                )
