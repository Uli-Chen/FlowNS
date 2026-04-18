"""Run FlowNeg experiment on a dataset with a backbone model."""
import argparse
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowneg  # noqa: F401 — applies RecBole compat patches

from recbole.config import Config
from recbole.data.utils import create_dataset
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.sampler import Sampler
from recbole.utils import init_seed, init_logger, set_color

from flowneg.flow_sampler import FlowNegSampler
from flowneg.flowneg_dataloader import FlowNegTrainDataLoader
from flowneg.flowneg_trainer import FlowNegTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--model', type=str, default=None, help='Override model name')
    parser.add_argument('--dataset', type=str, default=None, help='Override dataset')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    # Build config
    config_file_list = [args.config]
    if os.path.exists('configs/base.yaml'):
        config_file_list.insert(0, 'configs/base.yaml')

    config_dict = {}
    if args.seed is not None:
        config_dict['seed'] = args.seed

    model_name = args.model or 'BPR'
    config = Config(
        model=model_name,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    if args.dataset:
        config['dataset'] = args.dataset

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    # Create dataset
    dataset = create_dataset(config)
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = built_datasets

    # Samplers: standard for valid/test, FlowNeg for train
    standard_sampler = Sampler(
        ['train', 'valid', 'test'], built_datasets, 'uniform'
    )
    valid_sampler = standard_sampler.set_phase('valid')
    test_sampler = standard_sampler.set_phase('test')

    flowneg_sampler = FlowNegSampler(
        phases=['train', 'valid', 'test'],
        datasets=built_datasets,
        distribution='uniform',
        config=config,
    )
    train_sampler = flowneg_sampler.set_phase('train')

    # Dataloaders
    train_data = FlowNegTrainDataLoader(
        config, train_dataset, train_sampler, shuffle=True
    )
    valid_data = FullSortEvalDataLoader(
        config, valid_dataset, valid_sampler, shuffle=False
    )
    test_data = FullSortEvalDataLoader(
        config, test_dataset, test_sampler, shuffle=False
    )

    # Model
    # Import the correct model class based on config
    if model_name == 'LightGCN':
        from recbole.model.general_recommender.lightgcn import LightGCN
        model = LightGCN(config, dataset).to(config['device'])
    else:
        from recbole.model.general_recommender.bpr import BPR
        model = BPR(config, dataset).to(config['device'])

    # Trainer
    trainer = FlowNegTrainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=True
    )

    # Test
    test_result = trainer.evaluate(test_data, show_progress=True)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'model': model_name,
        'dataset': config['dataset'],
        'method': 'FlowNeg',
        'seed': config['seed'],
        'best_valid_score': float(best_valid_score) if best_valid_score else None,
        'best_valid_result': {k: float(v) for k, v in (best_valid_result or {}).items()},
        'test_result': {k: float(v) for k, v in (test_result or {}).items()},
        'timestamp': datetime.now().isoformat(),
        'config_file': args.config,
    }

    result_file = os.path.join(
        args.output_dir,
        f"flowneg_{model_name}_{config['dataset']}_{config['seed']}.json"
    )
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {result_file}")
    print(f"Best valid score: {best_valid_score}")
    print(f"Test result: {test_result}")


if __name__ == '__main__':
    main()
