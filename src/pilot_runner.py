"""Unified FlowNS experiment runner.

The shell entrypoint is scripts/run_pilot.sh. This module keeps the Python
training logic out of scripts/ so scripts stays small and stable.
"""
import argparse
import ast
import json
import logging
import os
from pathlib import Path

import torch
import yaml

from .custom_metrics import (
    compute_fn_rate,
    compute_w_statistics,
    theoretical_fn_bound,
)
from .flow_model import ConditionalFlowModel
from .flowns_trainer import FLOW_CONFIG_DEFAULTS, FlowNSTrainer
from .grpo import GRPOTrainer
from .neg_sampling import EmbeddingToItemMapper
from .recbole_utils import (
    get_embeddings,
    get_user_positive_items,
    setup_recbole,
    train_recbole,
)
from .reward import BoundaryAwareReward
from .sde_sampler import SDESampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_DIR / 'configs'
RESULTS_DIR = PROJECT_DIR / 'results' / 'pilot'
EXPERIMENTS_FILE = CONFIGS_DIR / 'experiments.yaml'

DATASET_CONFIGS = {
    'mind': CONFIGS_DIR / 'lightgcn_mind.yaml',
    'kuairand': CONFIGS_DIR / 'lightgcn_kuairand.yaml',
    'ml-100k': CONFIGS_DIR / 'lightgcn_ml100k.yaml',
    'yelp-2018': CONFIGS_DIR / 'lightgcn_yelp.yaml',
    'amazon-books': CONFIGS_DIR / 'lightgcn_amazon.yaml',
    'gowalla-merged': CONFIGS_DIR / 'lightgcn_gowalla.yaml',
}


def _safe_name(value):
    return value.replace('/', '_').replace('-', '_').replace('.', '_')


def _dataset_key_cache_path(path, dataset):
    """Isolate an embedding-space cache (flow / flow-ref) per dataset.

    A flow checkpoint lives in one recommender's embedding space, so a flow
    trained on dataset A must never be loaded for dataset B. The velocity-net
    state_dict shapes match across datasets (same embedding_size), so such a
    load would succeed silently but be semantically wrong. Insert the dataset
    name before the extension unless it is already present.
    """
    if not path:
        return path
    safe = _safe_name(dataset)
    p = Path(str(path))
    if f'_{safe}' in p.stem:
        return str(path)
    return str(p.with_name(f'{p.stem}_{safe}{p.suffix}'))


def _result_stem(experiment, dataset, seed=None, tag=None):
    stem = experiment
    if dataset != 'mind':
        stem = f'{stem}_{_safe_name(dataset)}'
    if seed is not None:
        stem = f'{stem}_s{int(seed)}'
    if tag:
        stem = f'{stem}_{_safe_name(str(tag))}'
    return stem


def _checkpoint_pointer(dataset, model):
    """Per (dataset, model) M0 pointer so backbone lines never clobber each other."""
    return RESULTS_DIR / (
        f'M0_model_path.{_safe_name(dataset)}.{_safe_name(model)}.txt'
    )


def _legacy_checkpoint_pointers(dataset):
    """Old pointer names (dataset-only) kept readable for existing checkpoints."""
    pointers = [RESULTS_DIR / f'M0_model_path.{_safe_name(dataset)}.txt']
    if dataset == 'mind':
        pointers.append(RESULTS_DIR / 'M0_model_path.txt')
    return pointers


def _load_m0_checkpoint_path(dataset, model):
    pointer_paths = [_checkpoint_pointer(dataset, model)]
    pointer_paths.extend(_legacy_checkpoint_pointers(dataset))

    pointer_path = next((path for path in pointer_paths if path.exists()), None)
    if pointer_path is None:
        searched = ', '.join(str(path) for path in pointer_paths)
        raise FileNotFoundError(
            f'M0 checkpoint pointer not found for dataset={dataset}, '
            f'model={model}. Searched: {searched}. Run the matching S0 '
            'baseline first.'
        )

    checkpoint_path = pointer_path.read_text().strip()
    if not checkpoint_path:
        raise ValueError(f'M0 checkpoint pointer is empty: {pointer_path}')

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'M0 checkpoint not found: {checkpoint_path}. Re-run the S0 baseline.'
        )
    return str(checkpoint_path)


def _save_m0_checkpoint_path(dataset, model, checkpoint_path):
    pointer_path = _checkpoint_pointer(dataset, model)
    pointer_path.write_text(str(checkpoint_path))
    logger.info('M0 checkpoint pointer saved to %s', pointer_path)


def _read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_value(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        lowered = value.lower()
        if lowered == 'true':
            return True
        if lowered == 'false':
            return False
        if lowered == 'none':
            return None
        return value


def _parse_overrides(overrides):
    parsed = {}
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'Override must be KEY=VALUE, got: {item}')
        key, value = item.split('=', 1)
        parsed[key] = _parse_value(value)
    return parsed


def _parse_phases(phases):
    if phases is None:
        return None
    return [phase.strip() for phase in phases.split(',') if phase.strip()]


def _dataset_config(dataset):
    if dataset not in DATASET_CONFIGS:
        valid = ', '.join(sorted(DATASET_CONFIGS))
        raise ValueError(f'Unsupported dataset: {dataset}. Valid datasets: {valid}')
    return DATASET_CONFIGS[dataset]


def _load_experiments():
    data = _read_yaml(EXPERIMENTS_FILE)
    recbole = data.get('recbole', {})
    defaults = data.get('defaults', {})
    experiments = data.get('experiments', {})
    return recbole, defaults, experiments


def _build_recbole_config_dict(recbole_base, dataset, overrides, seed=None):
    """Build RecBole config_dict: recbole_base < dataset_config < CLI overrides."""
    config_dict = dict(recbole_base)
    dataset_config = _read_yaml(_dataset_config(dataset))
    config_dict.update(dataset_config)
    config_dict.update(overrides)
    data_path = overrides.get('data_path', dataset_config.get('data_path'))
    if data_path is None:
        config_dict['data_path'] = str(PROJECT_DIR / 'data')
    else:
        data_path = Path(str(data_path))
        if not data_path.is_absolute():
            data_path = (PROJECT_DIR / data_path).resolve()
        config_dict['data_path'] = str(data_path)
    config_dict['dataset'] = dataset
    if seed is not None:
        config_dict['seed'] = seed
    return config_dict


def _resolve_experiment(experiment_name):
    """Load defaults + experiment overrides from experiments.yaml."""
    _, defaults, experiments = _load_experiments()
    if experiment_name not in experiments:
        valid = ', '.join(sorted(experiments))
        raise ValueError(
            f'Unknown experiment: {experiment_name}. '
            f'Valid experiments: {valid}'
        )
    merged = dict(defaults)
    exp = experiments[experiment_name]
    merged.update(exp)
    merged['experiment'] = experiment_name
    return merged


def _extract_flow_config(merged_config, overrides):
    flow_keys = set(FLOW_CONFIG_DEFAULTS)
    flow_config = {
        key: merged_config[key]
        for key in flow_keys
        if key in merged_config
    }
    for key, value in overrides.items():
        if key in flow_keys:
            flow_config[key] = value
    return flow_config


def _jsonable_result(result):
    if result is None:
        return None
    return {k: float(v) for k, v in result.items()}


def _run_quality_check(trainer, sample_users):
    trainer._ensure_flow_ready()
    user_emb, item_emb = get_embeddings(trainer.rec_model)
    mapper = EmbeddingToItemMapper(item_emb)
    all_pos = get_user_positive_items(trainer.dataset)

    sample_uids = list(trainer.user_pos_items.keys())[:sample_users]
    sample_uemb = user_emb[sample_uids]

    trainer.flow_model.velocity_net.eval()
    with torch.no_grad():
        final_emb, _, _ = trainer.sde_sampler.sample_trajectories(
            sample_uemb, n_trajectories=1,
        )
        final_emb = final_emb.squeeze(1)
        gen_item_ids = mapper.map_to_items(final_emb)

    fn_rate = compute_fn_rate(gen_item_ids, all_pos, sample_uids)
    logger.info(
        'Generation quality: users=%d, fn_rate=%.6f',
        len(sample_uids), fn_rate,
    )
    return {
        'fn_rate': float(fn_rate),
        'n_sample_users': len(sample_uids),
        'decision': 'PASS' if fn_rate < 0.05 else 'INVESTIGATE',
    }


def _run_generation_diagnostics(trainer, sample_users):
    if trainer.flow_model.ref_state_dict is None:
        return None

    user_emb, item_emb = get_embeddings(trainer.rec_model)
    trainer.mapper.update_embeddings(item_emb)
    all_pos = get_user_positive_items(trainer.dataset)

    sample_uids = [
        uid for uid in trainer.user_pos_items.keys() if uid < user_emb.shape[0]
    ][:sample_users]
    if not sample_uids:
        return None

    sample_uemb = user_emb[sample_uids]
    trainer.flow_model.velocity_net.eval()
    with torch.no_grad():
        final_emb, _, _ = trainer.sde_sampler.sample_trajectories(
            sample_uemb, n_trajectories=1,
        )
        final_emb = final_emb.squeeze(1)
        forbidden = [
            trainer.user_pos_items.get(int(uid), set()) for uid in sample_uids
        ]
        pos_embs, pos_mask = trainer.grpo_trainer._get_pos_embs(
            sample_uids, item_emb, trainer.user_pos_items,
        )
        mapped_ids = trainer.mapper.map_to_items(
            final_emb,
            forbidden_item_ids=forbidden,
            exclude_item_ids=(0,),
            user_emb=sample_uemb,
            pos_item_embs=pos_embs,
            pos_mask=pos_mask,
            reward_fn=trainer.reward_fn,
            strategy=trainer.mapping_strategy,
            candidate_topk=trainer.mapping_topk,
            boundary_safe_w=trainer.boundary_safe_w,
        )
        mapped_emb = item_emb[mapped_ids]

    fn_rate = compute_fn_rate(mapped_ids, all_pos, sample_uids)
    continuous_w = compute_w_statistics(sample_uemb, final_emb, pos_embs, pos_mask)
    mapped_w = compute_w_statistics(sample_uemb, mapped_emb, pos_embs, pos_mask)
    unique_ratio = float(mapped_ids.unique().numel() / max(mapped_ids.numel(), 1))

    logger.info(
        'Generation diagnostics: users=%d, fn_rate_all_known=%.6f, '
        'W_cont=%.4f, W_mapped=%.4f, unique_ratio=%.4f',
        len(sample_uids), fn_rate,
        continuous_w['mean'], mapped_w['mean'], unique_ratio,
    )
    return {
        'n_diagnostic_users': len(sample_uids),
        'fn_rate_all_known': float(fn_rate),
        'mapped_unique_ratio': unique_ratio,
        'w_continuous': continuous_w,
        'w_mapped': mapped_w,
    }


def _run_flowns_config(merged_config, dataset, seed=None):
    """Run one fully-merged FlowNS config and return (result_dict, trainer).

    This is the shared core used by both the CLI (run_experiment) and the grid
    search harness. It builds the RecBole + flow configs, runs every enabled
    phase, evaluates, and assembles the result dict. It performs no file IO and
    saves no checkpoints, so callers control persistence.
    """
    recbole_base, _, _ = _load_experiments()

    experiment = merged_config.get('experiment', 'adhoc')
    model_name = merged_config.get('model', 'LightGCN')
    recbole_flow_shared_keys = {'eval_step', 'stopping_step'}
    control_keys = (set(FLOW_CONFIG_DEFAULTS) - recbole_flow_shared_keys) | {
        'description',
        'experiment',
        'phases',
        'use_paired_m0',
        'evaluate_test',
        'quality_check',
        'quality_sample_users',
        'diagnostic_sample_users',
        'save_rec_checkpoint_pointer',
        'save_flow_model',
    }
    recbole_overrides = {
        key: value
        for key, value in merged_config.items()
        if key not in control_keys
    }

    config_dict = _build_recbole_config_dict(
        recbole_base, dataset, recbole_overrides, seed=seed,
    )
    config_dict['model'] = model_name

    logger.info('=== %s / dataset=%s / model=%s ===', experiment, dataset, model_name)

    config, model, ds, train_data, valid_data, test_data = setup_recbole(
        model_name, dataset,
        config_dict=config_dict,
    )

    flow_config = _extract_flow_config(merged_config, {})
    # Flow caches are embedding-space-specific: key them by dataset so a flow
    # pretrained on (e.g.) MIND is never silently loaded for KuaiRand.
    for _cache_key in ('flow_checkpoint_path', 'flow_ref_checkpoint_path'):
        if flow_config.get(_cache_key):
            flow_config[_cache_key] = _dataset_key_cache_path(
                flow_config[_cache_key], dataset,
            )
    if merged_config.get('use_paired_m0', False):
        flow_config['rec_checkpoint_path'] = _load_m0_checkpoint_path(
            dataset, model_name,
        )
        logger.info('Using paired M0 checkpoint: %s', flow_config['rec_checkpoint_path'])

    if 'enabled_phases' not in flow_config and 'phases' in merged_config:
        flow_config['enabled_phases'] = merged_config['phases']

    trainer = FlowNSTrainer(
        config, model, ds, train_data, valid_data, test_data,
        flow_config=flow_config,
    )
    best_score = trainer.run()

    test_result = None
    if merged_config.get('evaluate_test', True):
        test_result = trainer.evaluate()

    quality_result = None
    if merged_config.get('quality_check', False):
        quality_result = _run_quality_check(
            trainer, int(merged_config.get('quality_sample_users', 1000)),
        )

    diagnostic_result = _run_generation_diagnostics(
        trainer, int(merged_config.get('diagnostic_sample_users', 1000)),
    )

    result = {
        'experiment': experiment,
        'description': merged_config.get('description', ''),
        'dataset': dataset,
        'model': model_name,
        'seed': int(config['seed']),
        'enabled_phases': trainer.enabled_phases,
        'flow_config': flow_config,
        'best_valid_score': float(best_score) if best_score is not None else None,
        'test_result': _jsonable_result(test_result),
    }
    if quality_result:
        result.update(quality_result)
    if diagnostic_result:
        result['generation_diagnostics'] = diagnostic_result
    if trainer.grpo_trainer is not None and trainer.grpo_trainer.history:
        result['grpo_history'] = trainer.grpo_trainer.history
    return result, trainer


def run_experiment(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    merged_config = _resolve_experiment(args.experiment)
    overrides = _parse_overrides(args.set_values)
    phases = _parse_phases(args.phases)
    if phases is not None:
        overrides['enabled_phases'] = phases
    merged_config.update(overrides)

    result, trainer = _run_flowns_config(
        merged_config, args.dataset, seed=args.seed,
    )
    experiment = result['experiment']
    tag = getattr(args, 'tag', None)
    if tag:
        result['tag'] = tag
    stem = _result_stem(experiment, args.dataset, seed=args.seed, tag=tag)

    if merged_config.get('save_rec_checkpoint_pointer', False):
        _save_m0_checkpoint_path(
            args.dataset, result['model'], trainer._rec_trainer.saved_model_file,
        )

    if merged_config.get('save_flow_model', False):
        flow_path = RESULTS_DIR / f'{stem}_flow.pt'
        ref_path = RESULTS_DIR / f'{stem}_flow_ref.pt'
        torch.save(trainer.flow_model.velocity_net.state_dict(), flow_path)
        torch.save(trainer.flow_model.ref_state_dict, ref_path)
        logger.info('Flow checkpoint saved to %s', flow_path)

    out_path = RESULTS_DIR / f'{stem}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info('Results saved to %s', out_path)


def _measure_fn_rate(flow_model, sde_sampler, mapper, user_emb, user_pos, n_users):
    flow_model.velocity_net.eval()
    sample_uids = [u for u in user_pos if u < user_emb.shape[0]][:n_users]
    sample_uemb = user_emb[sample_uids]

    with torch.no_grad():
        final_emb, _, _ = sde_sampler.sample_trajectories(
            sample_uemb, n_trajectories=1,
        )
        final_emb = final_emb.squeeze(1)
        gen_ids = mapper.map_to_items(final_emb)

    return compute_fn_rate(gen_ids, user_pos, sample_uids)


def run_fn_guarantee(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    recbole_base, _, _ = _load_experiments()
    config_dict = _build_recbole_config_dict(
        recbole_base, args.dataset, {}, seed=args.seed,
    )

    logger.info('=== M5_fn_guarantee / dataset=%s ===', args.dataset)
    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', args.dataset,
        config_dict=config_dict,
    )
    train_recbole(config, model, train_data, valid_data)
    user_emb, item_emb = get_embeddings(model)
    user_pos = get_user_positive_items(dataset)

    emb_dim = config['embedding_size']
    flow = ConditionalFlowModel(
        emb_dim=emb_dim, hidden_dim=256, n_layers=3, device=str(config['device']),
    )
    flow.pretrain(
        user_emb, item_emb, user_pos,
        epochs=args.flow_epochs, lr=args.flow_lr,
    )

    sde = SDESampler(flow.velocity_net, n_steps=args.sde_steps, eta=args.eta, delta=args.delta)
    mapper = EmbeddingToItemMapper(item_emb)
    reward_fn = BoundaryAwareReward(a=1.0, gamma=1.0)

    fn_ref = _measure_fn_rate(flow, sde, mapper, user_emb, user_pos, args.fn_users)
    logger.info('Reference policy FN rate: %.6f', fn_ref)

    results_list = []
    for beta in args.beta_values:
        logger.info('--- beta=%s ---', beta)
        flow_beta = ConditionalFlowModel(
            emb_dim=emb_dim, hidden_dim=256, n_layers=3, device=str(config['device']),
        )
        flow_beta.velocity_net.load_state_dict(flow.ref_state_dict)
        flow_beta.ref_state_dict = {
            key: value.clone() for key, value in flow.ref_state_dict.items()
        }

        sde_beta = SDESampler(
            flow_beta.velocity_net,
            n_steps=args.sde_steps,
            eta=args.eta,
            delta=args.delta,
        )
        mapper_beta = EmbeddingToItemMapper(item_emb)
        grpo = GRPOTrainer(
            flow_beta, sde_beta, reward_fn, mapper_beta,
            group_size=8, clip_eps=0.2, beta=beta, lr=1e-4, max_pos_samples=10,
        )
        grpo.train(user_emb, item_emb, user_pos, epochs=args.grpo_epochs, batch_size=64)

        fn_actual = _measure_fn_rate(
            flow_beta, sde_beta, mapper_beta, user_emb, user_pos, args.fn_users,
        )
        fn_bound = theoretical_fn_bound(fn_ref, reward_fn.r_max, beta)
        tightness = (
            fn_actual / fn_bound
            if fn_bound > 0 and fn_bound != float('inf')
            else float('nan')
        )
        logger.info(
            'beta=%s: FN_actual=%.6f, FN_bound=%.6f, tightness=%.4f',
            beta, fn_actual, fn_bound, tightness,
        )
        results_list.append({
            'beta': beta,
            'fn_actual': fn_actual,
            'fn_bound': fn_bound,
            'tightness_ratio': tightness,
        })

    result = {
        'experiment': 'M5_fn_guarantee',
        'dataset': args.dataset,
        'fn_ref': fn_ref,
        'r_max': reward_fn.r_max,
        'results': results_list,
    }
    out_path = RESULTS_DIR / f'{_result_stem("M5_fn_guarantee", args.dataset)}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info('Results saved to %s', out_path)


def list_experiments(args):
    """Print available experiment names."""
    _, _, experiments = _load_experiments()
    for name, exp in experiments.items():
        desc = exp.get('description', '')
        phases = ', '.join(exp.get('phases', []))
        print(f'  {name:24s} phases=[{phases}]  {desc}')


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--experiment', required=True,
                            help='Experiment name from experiments.yaml')
    run_parser.add_argument('--dataset', default='mind')
    run_parser.add_argument('--seed', type=int, default=None)
    run_parser.add_argument('--phases', default=None,
                            help='Override phases, e.g. rec_pretrain,flow_pretrain')
    run_parser.add_argument('--set', dest='set_values', action='append', default=[],
                            help='Override KEY=VALUE; can be repeated')
    run_parser.add_argument('--tag', default=None,
                            help='Suffix for the result/flow file names; use to '
                                 'keep sweep runs from overwriting each other')
    run_parser.set_defaults(func=run_experiment)

    fn_parser = subparsers.add_parser('fn-guarantee')
    fn_parser.add_argument('--dataset', default='mind')
    fn_parser.add_argument('--seed', type=int, default=None)
    fn_parser.add_argument('--beta-values', type=float, nargs='+',
                           default=[0.01, 0.05, 0.1, 0.5, 1.0])
    fn_parser.add_argument('--flow-epochs', type=int, default=50)
    fn_parser.add_argument('--flow-lr', type=float, default=1e-4)
    fn_parser.add_argument('--grpo-epochs', type=int, default=5)
    fn_parser.add_argument('--sde-steps', type=int, default=20)
    fn_parser.add_argument('--eta', type=float, default=0.5)
    fn_parser.add_argument('--delta', type=float, default=0.01)
    fn_parser.add_argument('--fn-users', type=int, default=2000)
    fn_parser.set_defaults(func=run_fn_guarantee)

    list_parser = subparsers.add_parser('list')
    list_parser.set_defaults(func=list_experiments)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
