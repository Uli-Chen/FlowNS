"""Run a standard RecBole baseline (no FlowNeg) for comparison."""
import argparse
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowneg  # noqa: F401 — applies RecBole compat patches

from recbole.config import Config
from recbole.data.utils import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_model, get_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--model', type=str, default='BPR', help='Model name (e.g. BPR, LightGCN)')
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

    config = Config(
        model=args.model,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    if args.dataset:
        config['dataset'] = args.dataset

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    # Standard RecBole pipeline
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config['model'])(config, dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=True
    )

    test_result = trainer.evaluate(test_data, show_progress=True)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'model': args.model,
        'dataset': config['dataset'],
        'method': 'Baseline',
        'seed': config['seed'],
        'best_valid_score': float(best_valid_score) if best_valid_score else None,
        'best_valid_result': {k: float(v) for k, v in (best_valid_result or {}).items()},
        'test_result': {k: float(v) for k, v in (test_result or {}).items()},
        'timestamp': datetime.now().isoformat(),
        'config_file': args.config,
    }

    result_file = os.path.join(
        args.output_dir,
        f"baseline_{args.model}_{config['dataset']}_{config['seed']}.json"
    )
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {result_file}")
    print(f"Best valid score: {best_valid_score}")
    print(f"Test result: {test_result}")


if __name__ == '__main__':
    main()
