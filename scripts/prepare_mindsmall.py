#!/usr/bin/env python
"""Convert raw MINDsmall behaviours data to RecBole atomic (.inter) format
and build an exposure-negative cache for FlowNeg training.

Usage
-----
    python scripts/prepare_mindsmall.py \
        --data_dir  data_raw \
        --output_dir dataset/mindsmall \
        --cache_dir  exposure_cache
"""

import argparse
import os
import pickle
import statistics
from collections import defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_timestamp(ts_str):
    """Parse MIND timestamp string to Unix epoch float.

    Expected format: '11/13/2019 8:36:57 AM'
    """
    return datetime.strptime(ts_str, "%m/%d/%Y %I:%M:%S %p").timestamp()


def _parse_impressions(impressions_str):
    """Return (clicked_ids, non_clicked_ids) from an impressions field.

    Each entry looks like 'N4-1' (clicked) or 'N207-0' (not clicked).
    """
    clicked = []
    non_clicked = []
    for token in impressions_str.strip().split():
        if not token:
            continue
        parts = token.rsplit("-", 1)
        if len(parts) != 2:
            continue
        news_id, label = parts
        if label == "1":
            clicked.append(news_id)
        elif label == "0":
            non_clicked.append(news_id)
    return clicked, non_clicked


def parse_behaviors(filepath):
    """Parse a behaviors.tsv file.

    Returns
    -------
    interactions : list of (user_id, news_id, timestamp_float)
        One entry per clicked item in each impression.
    exposure_negs : dict  {user_id: set of non-clicked news_ids}
        Aggregated across all impressions for that user in this file.
    """
    interactions = []
    exposure_negs = defaultdict(set)

    with open(filepath, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                # Malformed line -- skip silently
                continue

            _imp_id, user_id, time_str, _history, impressions_str = cols[:5]

            try:
                ts = _parse_timestamp(time_str)
            except (ValueError, TypeError):
                # Unparseable timestamp -- skip
                continue

            clicked, non_clicked = _parse_impressions(impressions_str)

            for nid in clicked:
                interactions.append((user_id, nid, ts))

            for nid in non_clicked:
                exposure_negs[user_id].add(nid)

    return interactions, exposure_negs


# ---------------------------------------------------------------------------
# Write RecBole .inter
# ---------------------------------------------------------------------------

INTER_HEADER = "user_id:token\titem_id:token\ttimestamp:float\n"


def write_inter(records, path):
    """Write a list of (user_id, item_id, timestamp) to a RecBole .inter file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(INTER_HEADER)
        for uid, iid, ts in records:
            fh.write(f"{uid}\t{iid}\t{ts}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare MINDsmall data in RecBole .inter format."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/mindsmall",
        help="Root directory containing MINDsmall_train/ and MINDsmall_dev/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/mindsmall",
        help="Where to write .inter files (default: data/processed/mindsmall).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="exposure_cache",
        help="Where to write the exposure-neg pickle (default: exposure_cache).",
    )
    args = parser.parse_args()

    train_behaviors = os.path.join(
        args.data_dir, "MINDsmall_train", "behaviors.tsv"
    )
    dev_behaviors = os.path.join(
        args.data_dir, "MINDsmall_dev", "behaviors.tsv"
    )

    for path in (train_behaviors, dev_behaviors):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Expected file not found: {path}\n"
                "Run  python scripts/download_mind.py  first."
            )

    # ---- 1. Parse raw files ------------------------------------------------
    print("Parsing train behaviors ...")
    train_interactions, train_exposure_negs = parse_behaviors(train_behaviors)
    print(f"  {len(train_interactions)} clicked interactions from train")

    print("Parsing dev behaviors ...")
    dev_interactions, _ = parse_behaviors(dev_behaviors)
    print(f"  {len(dev_interactions)} clicked interactions from dev")

    # ---- 2. Temporal split of train data -----------------------------------
    # Sort train interactions by timestamp
    train_interactions.sort(key=lambda x: x[2])

    split_idx = int(len(train_interactions) * 0.8)
    train_split = train_interactions[:split_idx]
    valid_split = train_interactions[split_idx:]
    test_split = dev_interactions  # dev set becomes test

    print(f"\nTemporal split (train data):")
    print(f"  train : {len(train_split)}")
    print(f"  valid : {len(valid_split)}")
    print(f"  test  : {len(test_split)} (from dev set)")

    # ---- 3. Write .inter files ---------------------------------------------
    train_path = os.path.join(args.output_dir, "mindsmall.train.inter")
    valid_path = os.path.join(args.output_dir, "mindsmall.valid.inter")
    test_path = os.path.join(args.output_dir, "mindsmall.test.inter")

    write_inter(train_split, train_path)
    write_inter(valid_split, valid_path)
    write_inter(test_split, test_path)

    print(f"\nWrote .inter files to {args.output_dir}/")
    print(f"  {train_path}")
    print(f"  {valid_path}")
    print(f"  {test_path}")

    # ---- 4. Build exposure-negative cache ----------------------------------
    # Collect the set of positive (clicked) items per user in train data.
    train_pos = defaultdict(set)
    for uid, iid, _ in train_interactions:
        train_pos[uid].add(iid)

    # Remove any item that is actually positive for that user.
    exposure_cache = {}
    for uid, neg_set in train_exposure_negs.items():
        cleaned = neg_set - train_pos.get(uid, set())
        if cleaned:
            exposure_cache[uid] = cleaned

    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, "mindsmall_exposure_negs.pkl")
    with open(cache_path, "wb") as fh:
        pickle.dump(exposure_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nExposure-negative cache written to {cache_path}")

    # ---- 5. Statistics -----------------------------------------------------
    all_users = set()
    all_items = set()
    for records in (train_split, valid_split, test_split):
        for uid, iid, _ in records:
            all_users.add(uid)
            all_items.add(iid)

    neg_counts = [len(v) for v in exposure_cache.values()]

    print("\n===== Statistics =====")
    print(f"Total unique users : {len(all_users)}")
    print(f"Total unique items : {len(all_items)}")
    print(f"Train interactions : {len(train_split)}")
    print(f"Valid interactions : {len(valid_split)}")
    print(f"Test  interactions : {len(test_split)}")
    print(f"Users with exposure negs : {len(exposure_cache)}")
    if neg_counts:
        print(
            f"Exposure negs per user  : "
            f"min={min(neg_counts)}, "
            f"max={max(neg_counts)}, "
            f"mean={statistics.mean(neg_counts):.1f}, "
            f"median={statistics.median(neg_counts):.1f}"
        )
    print("======================")


if __name__ == "__main__":
    main()
