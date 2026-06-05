"""M4: Run ablation experiments."""
import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'RecBole')))

from src.recbole_utils import setup_recbole
from src.flowns_trainer import FlowNSTrainer

logging.basicConfig(level=logging.INFO)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')
BASE_CONFIG = os.path.join(CONFIGS_DIR, 'base.yaml')

DATASET_CONFIGS = {
    'yelp-2018': os.path.join(CONFIGS_DIR, 'lightgcn_yelp.yaml'),
    'amazon-books': os.path.join(CONFIGS_DIR, 'lightgcn_amazon.yaml'),
    'gowalla-merged': os.path.join(CONFIGS_DIR, 'lightgcn_gowalla.yaml'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='yelp-2018')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--ablation', required=True,
                        choices=['unshaped', 'no_rl', 'symmetric', 'raw_score',
                                 'beta_0', 'beta_sweep', 'gamma_sweep'])
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=1.0)
    args = parser.parse_args()

    config_files = [BASE_CONFIG]
    if args.dataset in DATASET_CONFIGS:
        config_files.append(DATASET_CONFIGS[args.dataset])

    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', args.dataset,
        config_file_list=config_files,
        config_dict={'seed': args.seed},
    )

    flow_config = {
        'flow_hidden_dim': 256,
        'flow_n_layers': 3,
        'flow_pretrain_epochs': 50,
        'sde_steps': 20,
        'grpo_epochs': 10,
        'grpo_group_size': 8,
        'grpo_beta': args.beta,
        'reward_a': 1.0,
        'reward_gamma': args.gamma,
        'joint_rec_epochs': 100,
        'joint_grpo_freq': 5,
        'joint_grpo_steps': 2,
        'eval_step': 5,
        'stopping_step': 10,
    }

    if args.ablation == 'unshaped':
        # R = W (no shaping, gamma effectively 0 with a=1)
        flow_config['reward_gamma'] = 0.0
        flow_config['reward_a'] = 1.0
    elif args.ablation == 'no_rl':
        flow_config['grpo_epochs'] = 0
        flow_config['joint_grpo_freq'] = 999999
    elif args.ablation == 'symmetric':
        flow_config['reward_gamma'] = 1.0
        flow_config['reward_a'] = 1.0
    elif args.ablation == 'raw_score':
        # Will need custom reward — handled in FlowNSTrainer
        flow_config['reward_type'] = 'raw_score'
    elif args.ablation == 'beta_0':
        flow_config['grpo_beta'] = 0.0
    elif args.ablation == 'beta_sweep':
        flow_config['grpo_beta'] = args.beta
    elif args.ablation == 'gamma_sweep':
        flow_config['reward_gamma'] = args.gamma

    trainer = FlowNSTrainer(
        config, model, dataset, train_data, valid_data, test_data,
        flow_config=flow_config,
    )

    trainer.run()
    result = trainer.evaluate()
    print(f'\n=== Ablation: {args.ablation} / {args.dataset} / seed={args.seed} ===')
    print(f'  beta={flow_config["grpo_beta"]}, gamma={flow_config["reward_gamma"]}')
    print(result)


if __name__ == '__main__':
    main()
