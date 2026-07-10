#!/usr/bin/env python3
"""Summarize FlowNS results/pilot/*.json into one comparison table.

Usage:
  python scripts/summarize_results.py [--dataset mind] [--baseline S0_m0_lgcn]
                                      [--filter S3_] [--sort ndcg]

Columns (see CLAUDE.md "Experiment focus"):
  realness: W_cont / W_map / FN / W>.8 (generation diagnostics, post-mask-fix)
  ranking:  valid + test NDCG@20 / Recall@20 (honest full-sort), Δ vs --baseline
"""
import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'pilot'


def _fmt(value, width=8, prec=4):
    if value is None:
        return '-'.rjust(width)
    if isinstance(value, str):
        return value.rjust(width)
    return f'{value:.{prec}f}'.rjust(width)


def _load_rows(dataset_filter, name_filter):
    rows = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict) or 'experiment' not in data:
            continue
        if dataset_filter and data.get('dataset') != dataset_filter:
            continue
        stem = path.stem
        if name_filter and name_filter not in stem:
            continue

        test = data.get('test_result') or {}
        diag = data.get('generation_diagnostics') or {}
        w_cont = (diag.get('w_continuous') or {}).get('mean')
        w_map = (diag.get('w_mapped') or {}).get('mean')
        w_map_hi = (diag.get('w_mapped') or {}).get('pct_above_0.8')
        fn = diag.get('fn_rate_all_known', data.get('fn_rate'))

        rows.append({
            'stem': stem,
            'model': data.get('model', '?'),
            'seed': data.get('seed'),
            'valid': data.get('best_valid_score'),
            'ndcg20': test.get('ndcg@20'),
            'recall20': test.get('recall@20'),
            'fn': fn,
            'w_cont': w_cont,
            'w_map': w_map,
            'w_map_hi': w_map_hi,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None,
                        help='Only rows whose result dataset matches')
    parser.add_argument('--baseline', default=None,
                        help='Result stem used as the Δ%% reference for NDCG@20')
    parser.add_argument('--filter', default=None,
                        help='Only stems containing this substring')
    parser.add_argument('--sort', choices=['name', 'ndcg'], default='name')
    args = parser.parse_args()

    rows = _load_rows(args.dataset, args.filter)
    if not rows:
        print(f'No result JSONs found under {RESULTS_DIR}')
        return

    if args.sort == 'ndcg':
        rows.sort(key=lambda r: (r['ndcg20'] is None, -(r['ndcg20'] or 0)))
    else:
        rows.sort(key=lambda r: r['stem'])

    base_ndcg = None
    if args.baseline:
        base = next((r for r in rows if r['stem'] == args.baseline), None)
        if base is None or base['ndcg20'] is None:
            print(f'(baseline {args.baseline} not found or has no test NDCG@20)')
        else:
            base_ndcg = base['ndcg20']

    name_w = max(len(r['stem']) for r in rows) + 1
    header = (
        f"{'run'.ljust(name_w)} {'seed'.rjust(5)} "
        f"{'valid'.rjust(8)} {'ndcg@20'.rjust(8)} {'Δ%'.rjust(7)} "
        f"{'rec@20'.rjust(8)} {'FN'.rjust(8)} "
        f"{'W_cont'.rjust(8)} {'W_map'.rjust(8)} {'W>.8'.rjust(6)}"
    )
    print(header)
    print('-' * len(header))
    for r in rows:
        delta = '-'.rjust(7)
        if base_ndcg and r['ndcg20'] is not None:
            delta = f'{100.0 * (r["ndcg20"] / base_ndcg - 1.0):+.2f}'.rjust(7)
        print(
            f"{r['stem'].ljust(name_w)} {str(r['seed'] or '-').rjust(5)} "
            f"{_fmt(r['valid'])} {_fmt(r['ndcg20'])} {delta} "
            f"{_fmt(r['recall20'])} {_fmt(r['fn'], prec=4)} "
            f"{_fmt(r['w_cont'])} {_fmt(r['w_map'])} "
            f"{_fmt(r['w_map_hi'], width=6, prec=2)}"
        )


if __name__ == '__main__':
    main()
