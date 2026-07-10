"""Probe per-user exposure volume on MIND and KuaiRand.

Motivation: a flat FlowNS gain may be a data-sufficiency problem. The flow learns
a per-user negative distribution; if most users have only a handful of exposed
negatives, there is little signal to fit and even less to GRPO-optimize.

For each dataset we count, per user:
  - positives        : clicked interactions (.inter rows)
  - exposed_negs     : shown-but-not-clicked (.exposed_neg rows) -> the flow's training signal
  - total exposure   : positives + exposed_negs

We print summary stats and draw, per dataset, a histogram of total exposure/user
(plus an exposed-neg/user overlay). Output: results/probe/exposure_hist.png
"""
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'probe')


def count_per_user(path):
    """Return {user_id: row_count}, skipping the RecBole header line."""
    counts = defaultdict(int)
    with open(path) as f:
        next(f, None)  # header
        for line in f:
            if not line.strip():
                continue
            uid = line.split('\t', 1)[0]
            counts[uid] += 1
    return counts


def summarize(name, arr):
    arr = np.asarray(arr, dtype=np.float64)
    pct = np.percentile(arr, [10, 25, 50, 75, 90, 95, 99])
    print(f'  {name:14s} n={len(arr):6d} mean={arr.mean():8.1f} '
          f'std={arr.std():8.1f} min={arr.min():.0f} max={arr.max():.0f}')
    print(f'  {"":14s} pcts  p10={pct[0]:.0f} p25={pct[1]:.0f} '
          f'p50={pct[2]:.0f} p75={pct[3]:.0f} p90={pct[4]:.0f} '
          f'p95={pct[5]:.0f} p99={pct[6]:.0f}')
    return arr


def load(name):
    inter = count_per_user(os.path.join(DATA, name, f'{name}.inter'))
    exposed = count_per_user(os.path.join(DATA, name, f'{name}.exposed_neg'))
    users = set(inter) | set(exposed)
    pos = np.array([inter.get(u, 0) for u in users])
    neg = np.array([exposed.get(u, 0) for u in users])
    total = pos + neg
    print(f'[{name}] users={len(users)} '
          f'total_pos={pos.sum()} total_exposed_neg={neg.sum()}')
    summarize('positives', pos)
    summarize('exposed_negs', neg)
    summarize('total_expo', total)
    print()
    return {'users': len(users), 'pos': pos, 'neg': neg, 'total': total}


def main():
    os.makedirs(OUT, exist_ok=True)
    datasets = ['mind', 'kuairand']
    data = {name: load(name) for name in datasets}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name in zip(axes, datasets):
        d = data[name]
        # cap the x-axis at p99 so the long tail doesn't flatten the bulk
        cap = np.percentile(d['total'], 99)
        bins = np.linspace(0, cap, 50)
        ax.hist(np.clip(d['neg'], 0, cap), bins=bins, alpha=0.6,
                label='exposed negs/user', color='#d6604d')
        ax.hist(np.clip(d['total'], 0, cap), bins=bins, alpha=0.45,
                label='total exposure/user', color='#4393c3')
        med = np.median(d['total'])
        ax.axvline(med, color='k', ls='--', lw=1,
                   label=f'median total={med:.0f}')
        ax.set_title(f'{name}  (n={d["users"]} users, x capped at p99={cap:.0f})')
        ax.set_xlabel('items per user')
        ax.set_ylabel('# users')
        ax.legend(fontsize=8)
    fig.suptitle('Per-user exposure distribution: MIND vs KuaiRand')
    fig.tight_layout()
    out = os.path.join(OUT, 'exposure_hist.png')
    fig.savefig(out, dpi=130)
    print(f'Saved: {os.path.relpath(out)}')


if __name__ == '__main__':
    main()
