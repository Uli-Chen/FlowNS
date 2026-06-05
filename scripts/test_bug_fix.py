"""Bug fix verification: Phase 1 + Phase 3 (random neg only, skip Phase 2).

If the bug is fixed, Phase 3 with random negatives should NOT meaningfully
outperform Phase 1, since both use uniform random negative sampling.
"""
import sys
import os
import json
import logging
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recbole_utils import (
    setup_recbole, get_embeddings, get_user_positive_items, _load_best_checkpoint,
)
from src.flowns_trainer import FlowNSTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'pilot')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config_files = [
        os.path.join(CONFIGS_DIR, 'base.yaml'),
        os.path.join(CONFIGS_DIR, 'lightgcn_mind.yaml'),
    ]
    config_dict = {
        'data_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')),
        'seed': 2020,
    }

    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', 'mind', config_file_list=config_files, config_dict=config_dict,
    )

    flow_config = {
        'flow_hidden_dim': 256,
        'flow_n_layers': 3,
        'flow_pretrain_epochs': 50,
        'flow_lr': 1e-4,
        'sde_steps': 20,
        'eta': 0.5,
        'delta': 0.01,
        'grpo_epochs': 0,
        'grpo_group_size': 8,
        'grpo_clip_eps': 0.2,
        'grpo_beta': 0.1,
        'grpo_lr': 1e-4,
        'grpo_batch_size': 64,
        'reward_a': 1.0,
        'reward_gamma': 1.0,
        'joint_rec_epochs': 50,
        'joint_grpo_freq': 999,  # effectively disable GRPO in Phase 3
        'joint_grpo_steps': 0,
        'eval_step': 5,
        'stopping_step': 10,
        'max_pos_samples': 10,
        'use_random_neg': True,
    }

    trainer = FlowNSTrainer(
        config, model, dataset, train_data, valid_data, test_data,
        flow_config=flow_config,
    )

    # === Phase 1 ===
    logger.info('========== Phase 1: RecBole pretraining ==========')
    best_score_p1, _ = trainer.phase1()

    # Evaluate Phase 1 best model (already loaded by our fix)
    logger.info('========== Evaluating Phase 1 best model ==========')
    result_p1 = trainer._rec_trainer.evaluate(
        test_data, load_best_model=False, show_progress=True,
    )
    logger.info(f'Phase 1 test result: {result_p1}')

    # === Phase 3 (skip Phase 2) ===
    logger.info('========== Phase 3: Joint training with RANDOM negatives ==========')
    best_score_p3 = trainer.phase3()

    # Evaluate Phase 3 best model
    logger.info('========== Evaluating Phase 3 best model ==========')
    result_p3 = trainer.evaluate()
    logger.info(f'Phase 3 test result: {result_p3}')

    # === Compare ===
    print('\n' + '=' * 60)
    print('  Bug Fix Verification Results')
    print('=' * 60)

    metrics = ['recall@20', 'ndcg@20', 'recall@10', 'ndcg@10']
    print(f'\n{"Metric":<15} {"Phase1":>10} {"Phase3-RNS":>12} {"Delta":>10}')
    print('-' * 50)
    for m in metrics:
        v1 = float(result_p1.get(m, 0))
        v3 = float(result_p3.get(m, 0))
        delta = v3 - v1
        sign = '+' if delta > 0 else ''
        print(f'{m:<15} {v1:>10.4f} {v3:>12.4f} {sign}{delta:>9.4f}')

    print('\nExpected: Phase 3 with random negatives should NOT significantly')
    print('outperform Phase 1 (both use uniform random negatives).')
    print('If Phase 3 >> Phase 1, the bug is NOT fixed.')
    print('=' * 60)

    results = {
        'experiment': 'bug_fix_verification',
        'phase1_valid': float(best_score_p1) if best_score_p1 is not None else None,
        'phase3_valid': float(best_score_p3) if best_score_p3 is not None else None,
        'phase1_test': {k: float(v) for k, v in result_p1.items()},
        'phase3_test': {k: float(v) for k, v in result_p3.items()},
    }
    out_path = os.path.join(RESULTS_DIR, 'bug_fix_verification.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
