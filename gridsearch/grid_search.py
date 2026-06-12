"""Grid search for the "flow, no GRPO" setting across RecBole backbones on MIND.

Each grid point runs a self-contained FlowNS pipeline (rec_pretrain ->
flow_pretrain -> joint, GRPO disabled) via src.pilot_runner._run_flowns_config,
and the harness writes an incremental JSON report into gridsearch/reports/.

Examples:
  # quick smoke: 1 epoch per phase, first grid point of each enabled backbone
  python -m gridsearch.grid_search --smoke

  # full sweep from the default space
  python -m gridsearch.grid_search

  # only some backbones, cap the number of runs
  python -m gridsearch.grid_search --backbones BPR,MultiVAE --max-runs 8

  # just print the plan (no training)
  python -m gridsearch.grid_search --list
"""
import argparse
import itertools
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

GRID_DIR = Path(__file__).resolve().parent
PROJECT_DIR = GRID_DIR.parent
REPORTS_DIR = GRID_DIR / 'reports'
DEFAULT_SPACE = GRID_DIR / 'search_space.yaml'

# Allow `python gridsearch/grid_search.py` as well as `-m gridsearch.grid_search`.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import yaml  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('gridsearch')

# Smoke overrides: make every phase do a single epoch so we only test plumbing.
SMOKE_OVERRIDES = {
    'epochs': 1,
    'flow_pretrain_epochs': 1,
    'joint_rec_epochs': 1,
    'diagnostic_sample_users': 200,
    'stopping_step': 100,
}


def _iter_combos(grid):
    """Yield one override dict per point in the cartesian product of `grid`."""
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def _summarize_test(test_result):
    if not isinstance(test_result, dict):
        return {}
    keys = ('recall@20', 'ndcg@20', 'recall@10', 'ndcg@10', 'hit@20', 'mrr@20')
    return {k: test_result[k] for k in keys if k in test_result}


def _summarize_diagnostics(diag):
    if not isinstance(diag, dict):
        return None
    out = {
        'fn_rate_all_known': diag.get('fn_rate_all_known'),
        'mapped_unique_ratio': diag.get('mapped_unique_ratio'),
    }
    for key in ('w_continuous', 'w_mapped'):
        block = diag.get(key)
        if isinstance(block, dict):
            out[f'{key}_mean'] = block.get('mean')
    return out


def _build_merged_config(space, name, backbone, combo, smoke):
    merged = {}
    merged.update(space.get('base', {}))
    merged.update(backbone.get('fixed', {}) or {})
    merged['model'] = backbone['model']
    merged.update(combo)
    merged['experiment'] = f'grid_{name}'
    if smoke:
        merged.update(SMOKE_OVERRIDES)
    return merged


def _planned_runs(space, backbones_filter, smoke):
    """Return list of (name, backbone_dict, combo) to execute."""
    runs = []
    for name, backbone in space.get('backbones', {}).items():
        if not backbone.get('enabled', False):
            continue
        if backbones_filter and name not in backbones_filter:
            continue
        combos = list(_iter_combos(backbone.get('grid', {})))
        if smoke:
            combos = combos[:1]  # one combo per backbone is enough to test plumbing
        for combo in combos:
            runs.append((name, backbone, combo))
    return runs


def _better(a, b):
    """Return the run dict with the higher best_valid_score (None = worst)."""
    av = a.get('best_valid_score') if a else None
    bv = b.get('best_valid_score') if b else None
    if bv is None:
        return a
    if av is None:
        return b
    return a if av >= bv else b


def run_grid(space, out_path, smoke=False, backbones_filter=None,
             max_runs=None, dataset_override=None, seed_override=None):
    from src.pilot_runner import _run_flowns_config

    dataset = dataset_override or space.get('dataset', 'mind')
    seed = seed_override if seed_override is not None else space.get('seed')

    planned = _planned_runs(space, backbones_filter, smoke)
    if max_runs is not None:
        planned = planned[:max_runs]

    report = {
        'dataset': dataset,
        'seed': seed,
        'smoke': smoke,
        'space_file': str(DEFAULT_SPACE.relative_to(PROJECT_DIR)),
        'started': datetime.now().isoformat(timespec='seconds'),
        'finished': None,
        'n_planned': len(planned),
        'n_ok': 0,
        'n_error': 0,
        'runs': [],
        'best_per_backbone': {},
    }

    def flush():
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2)

    logger.info('Grid search: dataset=%s seed=%s smoke=%s planned_runs=%d -> %s',
                dataset, seed, smoke, len(planned), out_path)
    flush()

    for idx, (name, backbone, combo) in enumerate(planned, 1):
        merged = _build_merged_config(space, name, backbone, combo, smoke)
        run = {
            'index': idx,
            'backbone': name,
            'model': backbone['model'],
            'experimental': bool(backbone.get('experimental', False)),
            'params': combo,
            'status': 'running',
            'best_valid_score': None,
            'test_result': None,
            'generation_diagnostics': None,
            'runtime_sec': None,
            'error': None,
        }
        report['runs'].append(run)
        flush()

        logger.info('[%d/%d] %s %s', idx, len(planned), name, combo)
        start = time.time()
        try:
            result, _trainer = _run_flowns_config(merged, dataset, seed=seed)
            run['status'] = 'ok'
            run['best_valid_score'] = result.get('best_valid_score')
            run['test_result'] = _summarize_test(result.get('test_result'))
            run['generation_diagnostics'] = _summarize_diagnostics(
                result.get('generation_diagnostics'))
            report['n_ok'] += 1
            prev = report['best_per_backbone'].get(name)
            report['best_per_backbone'][name] = {
                'params': combo,
                'best_valid_score': run['best_valid_score'],
                'test_result': run['test_result'],
            } if _better(run, prev) is run or prev is None else prev
            logger.info('[%d/%d] %s OK valid=%s test=%s', idx, len(planned), name,
                        run['best_valid_score'], run['test_result'])
        except Exception as exc:  # noqa: BLE001 - record and continue the sweep
            run['status'] = 'error'
            run['error'] = f'{type(exc).__name__}: {exc}'
            report['n_error'] += 1
            logger.error('[%d/%d] %s FAILED: %s', idx, len(planned), name, run['error'])
            logger.debug('%s', traceback.format_exc())
        finally:
            run['runtime_sec'] = round(time.time() - start, 1)
            flush()

    report['finished'] = datetime.now().isoformat(timespec='seconds')
    flush()
    _print_summary(report)
    return report


def _print_summary(report):
    print('\n==================== GRID SUMMARY ====================')
    print(f"dataset={report['dataset']} smoke={report['smoke']} "
          f"ok={report['n_ok']} error={report['n_error']} / {report['n_planned']}")
    for run in report['runs']:
        tag = run['status'].upper()
        valid = run['best_valid_score']
        ndcg = (run['test_result'] or {}).get('ndcg@20')
        extra = run['error'] if run['status'] == 'error' else f'valid={valid} ndcg@20={ndcg}'
        print(f"  [{run['backbone']:<9}] {tag:<5} {extra}  params={run['params']}")
    print('best per backbone:')
    for name, best in report['best_per_backbone'].items():
        print(f"  {name:<9} valid={best['best_valid_score']} "
              f"test={best['test_result']} params={best['params']}")
    print('=====================================================\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--space', default=str(DEFAULT_SPACE),
                        help='search space YAML (default: gridsearch/search_space.yaml)')
    parser.add_argument('--out', default=None, help='output JSON path (default: auto under reports/)')
    parser.add_argument('--smoke', action='store_true',
                        help='1 epoch per phase, first grid point per backbone')
    parser.add_argument('--backbones', default=None,
                        help='comma list to restrict, e.g. BPR,MultiVAE')
    parser.add_argument('--max-runs', type=int, default=None, help='cap total runs')
    parser.add_argument('--dataset', default=None, help='override dataset')
    parser.add_argument('--seed', type=int, default=None, help='override seed')
    parser.add_argument('--list', action='store_true', help='print the plan and exit')
    args = parser.parse_args()

    with open(args.space) as f:
        space = yaml.safe_load(f)

    backbones_filter = (
        {b.strip() for b in args.backbones.split(',') if b.strip()}
        if args.backbones else None
    )

    if args.list:
        planned = _planned_runs(space, backbones_filter, args.smoke)
        if args.max_runs is not None:
            planned = planned[:args.max_runs]
        print(f'Planned {len(planned)} runs (smoke={args.smoke}):')
        for i, (name, backbone, combo) in enumerate(planned, 1):
            print(f'  {i:3d}. {name:<9} {combo}')
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        dataset = args.dataset or space.get('dataset', 'mind')
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = 'smoke' if args.smoke else 'grid'
        out_path = REPORTS_DIR / f'{prefix}_{dataset}_{stamp}.json'

    run_grid(
        space, out_path, smoke=args.smoke, backbones_filter=backbones_filter,
        max_runs=args.max_runs, dataset_override=args.dataset, seed_override=args.seed,
    )


if __name__ == '__main__':
    main()
