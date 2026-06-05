"""Pilot M3b: RNS pretrain + RNS finetune control.
Tests whether the performance gain comes from flow-generated negatives
or merely from extra recommender training with random negatives.

This control skips flow pretraining and GRPO flow updates entirely.
"""
import sys
import os
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recbole_utils import setup_recbole
from src.flowns_trainer import FlowNSTrainer
from scripts.pilot_utils import load_m0_checkpoint_path

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

    logger.info('=== M3b: RNS pretrain + RNS finetune control ===')
    rec_checkpoint_path = load_m0_checkpoint_path(RESULTS_DIR)
    logger.info(f'Using paired M0 checkpoint: {rec_checkpoint_path}')

    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', 'mind', config_file_list=config_files, config_dict=config_dict,
    )

    flow_config = {
        'rec_checkpoint_path': rec_checkpoint_path,
        'joint_rec_epochs': 1000,
        'eval_step': 1,
        'stopping_step': 20,
        'use_random_neg': True,
    }

    trainer = FlowNSTrainer(
        config, model, dataset, train_data, valid_data, test_data,
        flow_config=flow_config,
    )

    best_score = trainer.run()
    test_result = trainer.evaluate()

    results = {
        'experiment': 'M3b_random_neg_control',
        'dataset': 'mind',
        'seed': 2020,
        'description': 'RNS pretrain + RNS finetune; skips flow pretrain and GRPO flow updates. '
                        'Controls for the effect of additional recommender training.',
        'use_random_neg': True,
        'paired_m0_checkpoint': rec_checkpoint_path,
        'best_valid_score': float(best_score) if best_score is not None else None,
        'test_result': {k: float(v) for k, v in test_result.items()},
    }
    out_path = os.path.join(RESULTS_DIR, 'M3b_random_neg_control.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
