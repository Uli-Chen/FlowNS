"""Pilot M3: Reward ablation on MIND.
Variants: Full (R=W(1-W)^γ), Unshaped (R=W), NoRL (no GRPO).
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

ABLATIONS = {
    'unshaped': {
        'reward_gamma': 0.0,
        'reward_a': 1.0,
        'description': 'R=W (no shaping, pure hardness maximization)',
    },
    'no_rl': {
        'grpo_epochs': 0,
        'joint_grpo_freq': 999999,
        'description': 'No GRPO, use pretrained flow model directly',
    },
}


def run_ablation(name, overrides):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config_files = [
        os.path.join(CONFIGS_DIR, 'base.yaml'),
        os.path.join(CONFIGS_DIR, 'lightgcn_mind.yaml'),
    ]
    config_dict = {
        'data_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')),
        'seed': 2020,
    }

    logger.info(f'=== M3 Ablation: {name} — {overrides.get("description", "")} ===')
    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', 'mind', config_file_list=config_files, config_dict=config_dict,
    )

    flow_config = dict(BASE_FLOW_CONFIG)
    for k, v in overrides.items():
        if k != 'description':
            flow_config[k] = v

    trainer = FlowNSTrainer(
        config, model, dataset, train_data, valid_data, test_data,
        flow_config=flow_config,
    )

    best_score = trainer.run()
    test_result = trainer.evaluate()

    results = {
        'experiment': f'M3_ablation_{name}',
        'dataset': 'mind',
        'ablation': name,
        'description': overrides.get('description', ''),
        'overrides': {k: v for k, v in overrides.items() if k != 'description'},
        'best_valid_score': float(best_score) if best_score is not None else None,
        'test_result': {k: float(v) for k, v in test_result.items()},
    }
    out_path = os.path.join(RESULTS_DIR, f'M3_ablation_{name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to {out_path}')

    return results


if __name__ == '__main__':
    for name, overrides in ABLATIONS.items():
        run_ablation(name, overrides)
