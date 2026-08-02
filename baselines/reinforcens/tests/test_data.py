from __future__ import annotations

from pathlib import Path

import numpy as np

from reinforcens.data import InteractionData, SPLIT_FILENAMES
from reinforcens.sampling import NegativeSampler


def write_tiny_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = {
        "train": [
            "0,1,1,event_click,0\n",
            "0,2|3|1,2,list_show,0\n",
            "0,6,3,event_click,1\n",
            "1,4,1,event_click,0\n",
            "1,5|7,2,list_show,0\n",
            "2,8,1,event_click,0\n",
        ],
        "validation": [
            "0,4,1,event_click,0\n",
            "0,5|7,2,list_show,0\n",
            "1,2,1,event_click,0\n",
            "1,3|6,2,list_show,0\n",
        ],
        "test": [
            "0,8,1,event_click,0\n",
            "0,0|9,2,list_show,0\n",
            "1,1,1,event_click,0\n",
            "1,8|9,2,list_show,0\n",
        ],
    }
    for split, values in rows.items():
        (root / SPLIT_FILENAMES[split]).write_text("".join(values), encoding="utf-8")
    return root


def test_load_clean_and_cache(tmp_path: Path) -> None:
    data_path = write_tiny_dataset(tmp_path / "data")
    cache = tmp_path / "cache.npz"
    data = InteractionData.load(data_path, num_users=3, num_items=10, cache_path=cache)
    assert data.train_items.tolist() == [1, 6, 4, 8]
    assert data.exposure_conflicts_removed == 1
    assert data.train_exposures.for_user(0).tolist() == [2, 3]
    assert not data.train_exposures.contains(np.array([0]), np.array([1]))[0]
    cached = InteractionData.load(
        data_path, num_users=3, num_items=10, cache_path=cache
    )
    np.testing.assert_array_equal(cached.train_users, data.train_users)
    np.testing.assert_array_equal(
        cached.train_exposures.keys, data.train_exposures.keys
    )


def test_evaluation_lists_and_rns_candidates(tmp_path: Path) -> None:
    data = InteractionData.load(write_tiny_dataset(tmp_path), num_users=3, num_items=10)
    evaluation = data.make_eval_candidates("validation", length=5, seed=7)
    assert evaluation.items.shape == (2, 5)
    assert evaluation.labels.sum(axis=1).tolist() == [1, 1]
    sampler = NegativeSampler(data, seed=4)
    users = np.array([0, 1], dtype=np.int32)
    candidates = sampler.rns_candidates(users, count=4, exposure_count=1)
    assert candidates.shape == (2, 4)
    assert data.train_exposures.contains(users, candidates[:, -1]).all()
    expanded = np.broadcast_to(users[:, None], candidates[:, :-1].shape)
    assert not data.train_clicks.contains(expanded, candidates[:, :-1]).any()
    assert not data.train_exposures.contains(expanded, candidates[:, :-1]).any()
