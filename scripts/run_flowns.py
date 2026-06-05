"""M2-M3: Run FlowNS full pipeline."""
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

DEFAULT_FLOW_CONFIG = {
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
    'joint_rec_epochs': 100,
    'joint_grpo_freq': 5,
    'joint_grpo_steps': 2,
    'eval_step': 5,
    'stopping_step': 10,
    'max_pos_samples': 10,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='yelp-2018',
                        choices=['yelp-2018', 'amazon-books', 'gowalla-merged', 'ml-100k'])
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--grpo_epochs', type=int, default=10)
    parser.add_argument('--joint_epochs', type=int, default=100)
    args = parser.parse_args()

    config_files = [BASE_CONFIG]
    if args.dataset in DATASET_CONFIGS:
        config_files.append(DATASET_CONFIGS[args.dataset])

    config_dict = {'seed': args.seed}
    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', args.dataset,
        config_file_list=config_files,
        config_dict=config_dict,
    )

    flow_config = dict(DEFAULT_FLOW_CONFIG)
    flow_config['reward_gamma'] = args.gamma
    flow_config['grpo_beta'] = args.beta
    flow_config['grpo_epochs'] = args.grpo_epochs
    flow_config['joint_rec_epochs'] = args.joint_epochs

    trainer = FlowNSTrainer(
        config, model, dataset, train_data, valid_data, test_data,
        flow_config=flow_config,
    )

    trainer.run()
    result = trainer.evaluate()
    print(f'\n=== FlowNS Result: {args.dataset} / seed={args.seed} / γ={args.gamma} / β={args.beta} ===')
    print(result)


if __name__ == '__main__':
    main()
