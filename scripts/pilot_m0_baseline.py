"""Pilot M0: Train LightGCN baseline on MIND dataset."""
import sys
import os
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recbole_utils import setup_recbole, train_recbole, evaluate_recbole

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

    logger.info('=== M0: LightGCN Baseline on MIND ===')
    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', 'mind',
        config_file_list=config_files,
        config_dict=config_dict,
    )

    trainer, best_score, best_result = train_recbole(config, model, train_data, valid_data)
    logger.info(f'Best valid score: {best_score}')
    logger.info(f'Best valid result: {best_result}')

    test_result = evaluate_recbole(trainer, test_data)
    logger.info(f'Test result: {test_result}')

    # Save results
    results = {
        'experiment': 'M0_baseline',
        'dataset': 'mind',
        'model': 'LightGCN',
        'neg_sampling': 'uniform',
        'seed': 2020,
        'best_valid_score': float(best_score),
        'test_result': {k: float(v) for k, v in test_result.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'M0_baseline.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to {RESULTS_DIR}/M0_baseline.json')

    # Save model checkpoint path for later phases
    saved_model = trainer.saved_model_file
    with open(os.path.join(RESULTS_DIR, 'M0_model_path.txt'), 'w') as f:
        f.write(saved_model)
    logger.info(f'Model saved at: {saved_model}')


if __name__ == '__main__':
    main()
