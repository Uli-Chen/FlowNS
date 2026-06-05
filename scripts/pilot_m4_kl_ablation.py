"""Pilot M4: KL (beta) ablation on MIND.
β ∈ {0, 0.1, 0.5}
"""
import sys
import os
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recbole_utils import setup_recbole
from src.flowns_trainer import FlowNSTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'pilot')

BASE_FLOW_CONFIG = {
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

BETA_VALUES = [0.0, 0.1, 0.5]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for beta in BETA_VALUES:
        config_files = [
            os.path.join(CONFIGS_DIR, 'base.yaml'),
            os.path.join(CONFIGS_DIR, 'lightgcn_mind.yaml'),
        ]
        config_dict = {
            'data_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')),
            'seed': 2020,
        }

        logger.info(f'=== M4 KL Ablation: β={beta} ===')
        config, model, dataset, train_data, valid_data, test_data = setup_recbole(
            'LightGCN', 'mind', config_file_list=config_files, config_dict=config_dict,
        )

        flow_config = dict(BASE_FLOW_CONFIG)
        flow_config['grpo_beta'] = beta

        trainer = FlowNSTrainer(
            config, model, dataset, train_data, valid_data, test_data,
            flow_config=flow_config,
        )

        best_score = trainer.run()
        test_result = trainer.evaluate()

        results = {
            'experiment': f'M4_kl_beta_{beta}',
            'dataset': 'mind',
            'beta': beta,
            'best_valid_score': float(best_score) if best_score is not None else None,
            'test_result': {k: float(v) for k, v in test_result.items()},
        }
        out_path = os.path.join(RESULTS_DIR, f'M4_kl_beta_{beta}.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
