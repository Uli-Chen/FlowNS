"""Pilot M2: Full FlowNS pipeline on MIND."""
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

    logger.info('=== M2: FlowNS Full Pipeline on MIND ===')
    rec_checkpoint_path = load_m0_checkpoint_path(RESULTS_DIR)
    logger.info(f'Using paired M0 checkpoint: {rec_checkpoint_path}')

    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', 'mind', config_file_list=config_files, config_dict=config_dict,
    )

    flow_config = {
        'rec_checkpoint_path': rec_checkpoint_path,
        'flow_hidden_dim': 256,
        'flow_n_layers': 3,
        'flow_pretrain_epochs': 50,
        'flow_lr': 1e-4,
        'sde_steps': 20,
        'eta': 0.5,
        'delta': 0.01,
        'grpo_epochs': 10,
        'grpo_group_size': 8,
        'grpo_clip_eps': 0.2,
        'grpo_beta': 0.1,
        'grpo_lr': 1e-4,
        'grpo_batch_size': 64,
        'reward_a': 1.0,
        'reward_gamma': 1.0,
        'joint_rec_epochs': 50,
        'joint_grpo_freq': 5,
        'joint_grpo_steps': 2,
        'eval_step': 5,
        'stopping_step': 10,
        'max_pos_samples': 10,
    }

    trainer = FlowNSTrainer(
        config, model, dataset, train_data, valid_data, test_data,
        flow_config=flow_config,
    )

    best_score = trainer.run()
    test_result = trainer.evaluate()

    results = {
        'experiment': 'M2_flowns_full',
        'dataset': 'mind',
        'seed': 2020,
        'paired_m0_checkpoint': rec_checkpoint_path,
        'flow_config': flow_config,
        'best_valid_score': float(best_score) if best_score is not None else None,
        'test_result': {k: float(v) for k, v in test_result.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'M2_flowns_full.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to {RESULTS_DIR}/M2_flowns_full.json')


if __name__ == '__main__':
    main()
