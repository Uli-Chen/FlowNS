"""Preprocess KuaiRand-1K into RecBole + FlowNS exposure format.

KuaiRand logs are impression-level: every row is a (user, video) the system
*exposed*, with binary feedback columns (is_click, long_view, is_like, ...).
This is exactly the exposure signal FlowNS needs, mirroring MINDsmall:

  - positive  (signal == 1): the user engaged  -> clicked interaction
  - exposed   (signal == 0): shown but not engaged -> *true* exposed negative

The raw video catalogue is huge and ultra-sparse (~4.4M videos for 1k users,
avg 2.7 impressions/video), so we apply a coarse item pre-filter by positive
count (`--min-item-inter`, default 10) purely for load tractability. The
canonical iterative k-core (on BOTH users and items) is applied by the RecBole
config (configs/lightgcn_kuairand.yaml: *_inter_num_interval). The pre-filter
is lossless w.r.t. an item k-core whenever min-item-inter <= k, because an item
needs >= k raw positive impressions to reach >= k distinct-user interactions.

Output (default data/kuairand/):
  kuairand.inter        — positive interactions (user_id, item_id, timestamp)
  kuairand.exposed_neg  — exposed negatives (user_id, item_id)
  kuairand.user2id.json — raw user_id (str) -> internal 1-based ID
  kuairand.item2id.json — raw video_id (str) -> internal 1-based ID

Run from the project root:
  python scripts/preprocess_kuairand.py --min-item-inter 10
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SRC_DIR = os.path.join(DATA_DIR, 'KuaiRand-1K', 'data')
STANDARD_LOGS = [
    'log_standard_4_08_to_4_21_1k.csv',
    'log_standard_4_22_to_5_08_1k.csv',
]
RANDOM_LOG = 'log_random_4_22_to_5_08_1k.csv'

USER_COL = 'user_id'
ITEM_COL = 'video_id'
TIME_COL = 'time_ms'
CHUNK = 2_000_000


def _log_paths(include_random):
    paths = [os.path.join(SRC_DIR, name) for name in STANDARD_LOGS]
    if include_random:
        paths.append(os.path.join(SRC_DIR, RANDOM_LOG))
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'KuaiRand log(s) not found: {missing}')
    return paths


def count_item_positives(paths, signal):
    """Pass 1: positive (signal==1) count per video, streamed in chunks."""
    counts = defaultdict(int)
    usecols = [ITEM_COL, signal]
    dtype = {ITEM_COL: 'int64', signal: 'int8'}
    for path in paths:
        for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=CHUNK):
            pos = chunk.loc[chunk[signal] == 1, ITEM_COL].value_counts()
            for vid, c in pos.items():
                counts[int(vid)] += int(c)
    return counts


def collect_interactions(paths, signal, kept_items):
    """Pass 2: per-user positive {item: earliest_ts} and exposed-negative items."""
    pos = defaultdict(dict)       # user -> {item: min timestamp}
    neg = defaultdict(set)        # user -> {item}
    usecols = [USER_COL, ITEM_COL, TIME_COL, signal]
    dtype = {USER_COL: 'int32', ITEM_COL: 'int64', TIME_COL: 'int64', signal: 'int8'}
    for path in paths:
        for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=CHUNK):
            chunk = chunk[chunk[ITEM_COL].isin(kept_items)]
            for u, i, t, s in chunk.itertuples(index=False, name=None):
                u, i = int(u), int(i)
                if s == 1:
                    prev = pos[u].get(i)
                    if prev is None or t < prev:
                        pos[u][i] = int(t)
                else:
                    neg[u].add(i)
    return pos, neg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signal', default='is_click',
                        help='feedback column treated as positive (default: is_click)')
    parser.add_argument('--min-item-inter', type=int, default=10,
                        help='coarse item pre-filter: keep videos with >= this many raw '
                             'positives (default: 10). Keep <= the RecBole config k-core to '
                             'stay lossless; canonical iterative k-core lives in the config.')
    parser.add_argument('--min-user-inter', type=int, default=10,
                        help='coarse user pre-filter: drop users with fewer positives after '
                             'the item pre-filter (default: 10). Canonical core is in the config.')
    parser.add_argument('--include-random', action='store_true',
                        help='also fold in the unbiased log_random file')
    parser.add_argument('--name', default='kuairand', help='output dataset name')
    parser.add_argument('--out', default=None, help='output dir (default data/<name>)')
    args = parser.parse_args()

    out_dir = args.out or os.path.join(DATA_DIR, args.name)
    os.makedirs(out_dir, exist_ok=True)
    paths = _log_paths(args.include_random)

    print(f'Pass 1/2: counting "{args.signal}" positives per video over {len(paths)} log(s)...')
    item_pos_count = count_item_positives(paths, args.signal)
    kept_items = {v for v, c in item_pos_count.items() if c >= args.min_item_inter}
    print(f'  videos with >=1 positive: {len(item_pos_count)}; '
          f'kept (>= {args.min_item_inter}): {len(kept_items)}')
    if not kept_items:
        raise SystemExit('No items survive the item filter; lower --min-item-inter.')

    print('Pass 2/2: collecting per-user positives and exposed negatives...')
    pos, neg = collect_interactions(paths, args.signal, kept_items)

    # Drop low-activity users (k-core on the user side); all KuaiRand-1K users
    # are heavy, so this is a safety net rather than an active filter.
    kept_users = {u for u, items in pos.items() if len(items) >= args.min_user_inter}
    print(f'  users with positives: {len(pos)}; kept (>= {args.min_user_inter}): {len(kept_users)}')

    # Materialise unique positives; an exposed negative that is also a positive
    # for the same user is a re-impression of a liked item -> not a negative.
    clicks = []  # (user, item, timestamp)
    for u in kept_users:
        for i, t in pos[u].items():
            clicks.append((u, i, t))
    exposed = []  # (user, item)
    for u in kept_users:
        pos_items = pos[u]
        for i in neg[u]:
            if i not in pos_items:
                exposed.append((u, i))

    print(f'Unique positives: {len(clicks)}, exposed negatives: {len(exposed)}')

    # 1-based contiguous IDs (0 reserved for RecBole PAD), sorted for determinism.
    users = sorted({u for u, _, _ in clicks})
    items = sorted({i for _, i, _ in clicks})
    user2id = {str(u): idx + 1 for idx, u in enumerate(users)}
    item2id = {str(i): idx + 1 for idx, i in enumerate(items)}
    print(f'Users: {len(user2id)}, Items: {len(item2id)}')

    inter_path = os.path.join(out_dir, f'{args.name}.inter')
    with open(inter_path, 'w') as f:
        f.write('user_id:token\titem_id:token\ttimestamp:float\n')
        for u, i, t in clicks:
            f.write(f'{user2id[str(u)]}\t{item2id[str(i)]}\t{t}\n')
    print(f'Written: {inter_path}')

    exposed_path = os.path.join(out_dir, f'{args.name}.exposed_neg')
    with open(exposed_path, 'w') as f:
        f.write('user_id:token\titem_id:token\n')
        for u, i in exposed:
            # item already guaranteed in item2id (kept item with positives)
            f.write(f'{user2id[str(u)]}\t{item2id[str(i)]}\n')
    print(f'Written: {exposed_path}')

    with open(os.path.join(out_dir, f'{args.name}.user2id.json'), 'w') as f:
        json.dump(user2id, f)
    with open(os.path.join(out_dir, f'{args.name}.item2id.json'), 'w') as f:
        json.dump(item2id, f)

    # Stats
    n_users, n_items = len(user2id), len(item2id)
    avg_pos = len(clicks) / max(n_users, 1)
    avg_neg = len(exposed) / max(n_users, 1)
    density = len(clicks) / max(n_users * n_items, 1)
    print(f'Avg positives/user: {avg_pos:.1f}, avg exposed negs/user: {avg_neg:.1f}')
    print(f'Interaction-matrix density: {density:.4%}')


if __name__ == '__main__':
    main()
