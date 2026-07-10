"""Preprocess MINDsmall into RecBole-compatible format.

MINDsmall has exposure data: each impression is (news_id, 0/1).
- label=1: clicked (positive interaction)
- label=0: exposed but not clicked (true negative)

Output:
  data/mind/mind.inter        — clicked interactions (user, item, timestamp)
  data/mind/mind.exposed_neg  — exposed negatives (user, item)
  data/mind/mind.user2id.json — user string → internal ID
  data/mind/mind.item2id.json — item string → internal ID
"""
import os
import sys
import json
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'MINDsmall_train')
DEV_DIR = os.path.join(DATA_DIR, 'MINDsmall_dev', 'MINDsmall_dev')
OUT_DIR = os.path.join(DATA_DIR, 'mind')


def parse_behaviors(filepath):
    """Parse behaviors.tsv → (clicks, exposed_negs, histories).

    Returns:
        clicks: list of (user_str, item_str, timestamp_str)
        exposed_negs: list of (user_str, item_str)
    """
    clicks = []
    exposed_negs = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            uid = parts[1]
            timestamp = parts[2]

            for imp in parts[4].split():
                sep = imp.rfind('-')
                nid = imp[:sep]
                label = imp[sep + 1:]
                if label == '1':
                    clicks.append((uid, nid, timestamp))
                else:
                    exposed_negs.append((uid, nid))

    return clicks, exposed_negs


def build_id_maps(clicks, exposed_negs):
    users = set()
    items = set()
    for u, i, _ in clicks:
        users.add(u)
        items.add(i)
    for u, i in exposed_negs:
        users.add(u)
        items.add(i)

    user2id = {u: idx + 1 for idx, u in enumerate(sorted(users))}
    item2id = {i: idx + 1 for idx, i in enumerate(sorted(items))}
    return user2id, item2id


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Parsing train behaviors...')
    clicks_train, exposed_negs_train = parse_behaviors(
        os.path.join(TRAIN_DIR, 'behaviors.tsv')
    )
    print(f'  Train: {len(clicks_train)} clicks, {len(exposed_negs_train)} exposed negs')

    dev_path = os.path.join(DEV_DIR, 'behaviors.tsv')
    clicks_dev, exposed_negs_dev = [], []
    if os.path.exists(dev_path):
        print('Parsing dev behaviors...')
        clicks_dev, exposed_negs_dev = parse_behaviors(dev_path)
        print(f'  Dev: {len(clicks_dev)} clicks, {len(exposed_negs_dev)} exposed negs')

    all_clicks = clicks_train + clicks_dev
    all_exposed = exposed_negs_train + exposed_negs_dev

    # Deduplicate clicks per (user, item) — keep first occurrence
    seen_clicks = set()
    unique_clicks = []
    for u, i, t in all_clicks:
        if (u, i) not in seen_clicks:
            seen_clicks.add((u, i))
            unique_clicks.append((u, i, t))

    # Deduplicate exposed negs
    unique_exposed = list(set(all_exposed))
    # Remove exposed negs that are actually clicks
    unique_exposed = [(u, i) for u, i in unique_exposed if (u, i) not in seen_clicks]

    print(f'Unique clicks: {len(unique_clicks)}, unique exposed negs: {len(unique_exposed)}')

    user2id, item2id = build_id_maps(unique_clicks, unique_exposed)
    print(f'Users: {len(user2id)}, Items: {len(item2id)}')

    # Write RecBole .inter file
    inter_path = os.path.join(OUT_DIR, 'mind.inter')
    with open(inter_path, 'w') as f:
        f.write('user_id:token\titem_id:token\ttimestamp:token\n')
        for u, i, t in unique_clicks:
            f.write(f'{user2id[u]}\t{item2id[i]}\t{t}\n')
    print(f'Written: {inter_path}')

    # Write exposed negatives
    exposed_path = os.path.join(OUT_DIR, 'mind.exposed_neg')
    with open(exposed_path, 'w') as f:
        f.write('user_id:token\titem_id:token\n')
        for u, i in unique_exposed:
            f.write(f'{user2id[u]}\t{item2id[i]}\n')
    print(f'Written: {exposed_path}')

    # Write ID maps
    with open(os.path.join(OUT_DIR, 'mind.user2id.json'), 'w') as f:
        json.dump(user2id, f)
    with open(os.path.join(OUT_DIR, 'mind.item2id.json'), 'w') as f:
        json.dump(item2id, f)

    # Stats
    user_click_count = defaultdict(int)
    for u, i, _ in unique_clicks:
        user_click_count[u] += 1
    user_exposed_count = defaultdict(int)
    for u, i in unique_exposed:
        user_exposed_count[u] += 1

    avg_clicks = sum(user_click_count.values()) / max(len(user_click_count), 1)
    avg_exposed = sum(user_exposed_count.values()) / max(len(user_exposed_count), 1)
    print(f'Avg clicks/user: {avg_clicks:.1f}, Avg exposed negs/user: {avg_exposed:.1f}')


if __name__ == '__main__':
    main()
