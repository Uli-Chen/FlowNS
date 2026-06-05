"""M1: Run RecBole baselines (LightGCN, BPR, Popularity, DNS)."""
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'RecBole')))

from recbole.quick_start import run_recbole


CONFIGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')
BASE_CONFIG = os.path.join(CONFIGS_DIR, 'base.yaml')

DATASET_CONFIGS = {
    'yelp-2018': os.path.join(CONFIGS_DIR, 'lightgcn_yelp.yaml'),
    'amazon-books': os.path.join(CONFIGS_DIR, 'lightgcn_amazon.yaml'),
    'gowalla-merged': os.path.join(CONFIGS_DIR, 'lightgcn_gowalla.yaml'),
}


def run_baseline(model, dataset, neg_sampling='uniform', seed=2020):
    config_files = [BASE_CONFIG]
    if dataset in DATASET_CONFIGS and model == 'LightGCN':
        config_files.append(DATASET_CONFIGS[dataset])

    config_dict = {'seed': seed}

    if model != 'LightGCN':
        config_dict['model'] = model

    if neg_sampling == 'popularity':
        config_dict['train_neg_sample_args'] = {
            'distribution': 'popularity',
            'sample_num': 1,
        }
    elif neg_sampling == 'dns':
        config_dict['train_neg_sample_args'] = {
            'distribution': 'uniform',
            'sample_num': 1,
            'dynamic': True,
            'candidate_num': 50,
        }

    result = run_recbole(
        model=model,
        dataset=dataset,
        config_file_list=config_files,
        config_dict=config_dict,
    )
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='LightGCN', choices=['LightGCN', 'BPR'])
    parser.add_argument('--dataset', default='yelp-2018',
                        choices=['yelp-2018', 'amazon-books', 'gowalla-merged', 'ml-100k'])
    parser.add_argument('--neg', default='uniform', choices=['uniform', 'popularity', 'dns'])
    parser.add_argument('--seed', type=int, default=2020)
    args = parser.parse_args()

    result = run_baseline(args.model, args.dataset, args.neg, args.seed)
    print(f'\n=== Result: {args.model} / {args.dataset} / {args.neg} / seed={args.seed} ===')
    print(result)
