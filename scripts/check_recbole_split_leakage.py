"""Diagnose whether a RecBole baseline model contains held-out interactions.

This check intentionally avoids training. It builds the configured RecBole
dataset/splits and inspects the interactions captured by the model at
construction time, where LightGCN builds its graph and MultiVAE builds the
history matrix.
"""
import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.flowns_trainer import FLOW_CONFIG_DEFAULTS
from src.pilot_runner import _build_recbole_config_dict, _load_experiments, _resolve_experiment
from src.recbole_utils import get_user_positive_items, setup_recbole

logging.basicConfig(level=logging.WARNING)


RECBOLE_FLOW_SHARED_KEYS = {'eval_step', 'stopping_step'}

CONTROL_KEYS = (set(FLOW_CONFIG_DEFAULTS) - RECBOLE_FLOW_SHARED_KEYS) | {
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


def _merge_pos(*pos_maps):
    merged = {}
    for pos_map in pos_maps:
        for uid, items in pos_map.items():
            merged.setdefault(uid, set()).update(items)
    return merged


def _count_pairs(pos_map):
    return sum(len(items) for items in pos_map.values())


def _count_overlap(source_pos, target_pos):
    users = 0
    pairs = 0
    for uid, target_items in target_pos.items():
        overlap = source_pos.get(uid, set()) & target_items
        if overlap:
            users += 1
            pairs += len(overlap)
    return {'users': users, 'pairs': pairs}


def _model_positive_items(model):
    if hasattr(model, 'history_item_id') and hasattr(model, 'history_item_value'):
        item_ids = model.history_item_id.detach().cpu()
        values = model.history_item_value.detach().cpu()
        pos = {}
        for uid in range(item_ids.shape[0]):
            mask = values[uid] != 0
            items = {int(i) for i in item_ids[uid][mask].tolist()}
            if items:
                pos[uid] = items
        return pos

    if hasattr(model, 'interaction_matrix'):
        matrix = model.interaction_matrix.tocoo()
        pos = {}
        for uid, iid in zip(matrix.row, matrix.col):
            pos.setdefault(int(uid), set()).add(int(iid))
        return pos

    raise ValueError(
        f'Unsupported model for leakage inspection: {model.__class__.__name__}'
    )


def _build_config(dataset, experiment, seed):
    recbole_base, _, _ = _load_experiments()
    merged = _resolve_experiment(experiment)
    model_name = merged.get('model', 'LightGCN')
    recbole_overrides = {
        key: value for key, value in merged.items() if key not in CONTROL_KEYS
    }
    config_dict = _build_recbole_config_dict(
        recbole_base, dataset, recbole_overrides, seed=seed,
    )
    config_dict['model'] = model_name
    config_dict['show_progress'] = False
    return model_name, config_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-100k')
    parser.add_argument('--experiment', default='M0_multivae_baseline')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    model_name, config_dict = _build_config(args.dataset, args.experiment, args.seed)
    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        model_name, args.dataset, config_dict=config_dict,
    )

    full_pos = get_user_positive_items(dataset)
    train_pos = get_user_positive_items(train_data)
    valid_pos = get_user_positive_items(valid_data)
    test_pos = get_user_positive_items(test_data)
    heldout_pos = _merge_pos(valid_pos, test_pos)
    model_pos = _model_positive_items(model)

    report = {
        'dataset': args.dataset,
        'experiment': args.experiment,
        'model': model_name,
        'seed': int(config['seed']),
        'data_path': str(config['data_path']),
        'split_unique_pairs': {
            'full': _count_pairs(full_pos),
            'train': _count_pairs(train_pos),
            'valid': _count_pairs(valid_pos),
            'test': _count_pairs(test_pos),
            'heldout': _count_pairs(heldout_pos),
            'model_constructor_state': _count_pairs(model_pos),
        },
        'overlap_with_heldout_unique_pairs': {
            'old_full_dataset_constructor_would_have': _count_overlap(full_pos, heldout_pos),
            'train_split_only_should_have': _count_overlap(train_pos, heldout_pos),
            'current_model_constructor_has': _count_overlap(model_pos, heldout_pos),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
