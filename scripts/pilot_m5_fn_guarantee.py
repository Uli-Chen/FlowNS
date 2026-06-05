"""Pilot M5: FN guarantee verification on MIND.
Uses MIND's exposure data to compute true FN rate vs theoretical bound.
"""
import sys
import os
import json
import logging
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recbole_utils import setup_recbole, train_recbole, get_embeddings, get_user_positive_items
from src.flow_model import ConditionalFlowModel
from src.sde_sampler import SDESampler
from src.reward import BoundaryAwareReward
from src.grpo import GRPOTrainer
from src.neg_sampling import EmbeddingToItemMapper
from src.custom_metrics import compute_fn_rate, theoretical_fn_bound

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'pilot')

BETA_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]


def load_exposed_negatives():
    """Load exposed negative items from MIND data as ground-truth negatives."""
    exposed_neg = {}
    path = os.path.join(DATA_DIR, 'mind', 'mind.exposed_neg')
    with open(path, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            uid = int(parts[0])
            iid = int(parts[1])
            if uid not in exposed_neg:
                exposed_neg[uid] = set()
            exposed_neg[uid].add(iid)
    return exposed_neg


def measure_fn_rate(flow_model, sde_sampler, mapper, user_emb, user_pos, n_users=2000):
    """Generate negatives and measure FN rate against positive items."""
    flow_model.velocity_net.eval()
    sample_uids = [u for u in user_pos if u < user_emb.shape[0]][:n_users]
    sample_uemb = user_emb[sample_uids]

    with torch.no_grad():
        final_emb, _, _ = sde_sampler.sample_trajectories(sample_uemb, n_trajectories=1)
        final_emb = final_emb.squeeze(1)
        gen_ids = mapper.map_to_items(final_emb)

    fn_rate = compute_fn_rate(gen_ids, user_pos, sample_uids)
    return fn_rate


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config_files = [
        os.path.join(CONFIGS_DIR, 'base.yaml'),
        os.path.join(CONFIGS_DIR, 'lightgcn_mind.yaml'),
    ]
    config_dict = {
        'data_path': os.path.abspath(os.path.join(DATA_DIR)),
        'seed': 2020,
    }

    logger.info('=== M5: FN Guarantee Verification on MIND ===')

    # Step 1: Train LightGCN
    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', 'mind', config_file_list=config_files, config_dict=config_dict,
    )
    trainer, best_score, _ = train_recbole(config, model, train_data, valid_data)
    user_emb, item_emb = get_embeddings(model)
    user_pos = get_user_positive_items(dataset)
    device = config['device']

    # Step 2: Pretrain flow model (π_ref)
    emb_dim = config['embedding_size']
    flow = ConditionalFlowModel(emb_dim=emb_dim, hidden_dim=256, n_layers=3, device=str(device))
    flow.pretrain(user_emb, item_emb, user_pos, epochs=50, lr=1e-4)

    sde = SDESampler(flow.velocity_net, n_steps=20, eta=0.5, delta=0.01)
    mapper = EmbeddingToItemMapper(item_emb)
    reward_fn = BoundaryAwareReward(a=1.0, gamma=1.0)

    # Step 3: Measure reference FN rate
    fn_ref = measure_fn_rate(flow, sde, mapper, user_emb, user_pos)
    logger.info(f'Reference policy FN rate: {fn_ref:.6f}')

    # Step 4: For each β, GRPO fine-tune and measure FN rate
    results_list = []
    for beta in BETA_VALUES:
        logger.info(f'\n--- β = {beta} ---')

        # Fresh flow model from reference
        flow_beta = ConditionalFlowModel(emb_dim=emb_dim, hidden_dim=256, n_layers=3, device=str(device))
        flow_beta.velocity_net.load_state_dict(flow.ref_state_dict)
        flow_beta.ref_state_dict = {k: v.clone() for k, v in flow.ref_state_dict.items()}

        sde_beta = SDESampler(flow_beta.velocity_net, n_steps=20, eta=0.5, delta=0.01)
        mapper_beta = EmbeddingToItemMapper(item_emb)

        grpo = GRPOTrainer(
            flow_beta, sde_beta, reward_fn, mapper_beta,
            group_size=8, clip_eps=0.2, beta=beta, lr=1e-4, max_pos_samples=10,
        )
        grpo.train(user_emb, item_emb, user_pos, epochs=5, batch_size=64)

        fn_actual = measure_fn_rate(flow_beta, sde_beta, mapper_beta, user_emb, user_pos)
        fn_bound = theoretical_fn_bound(fn_ref, reward_fn.r_max, beta)
        tightness = fn_actual / fn_bound if fn_bound > 0 and fn_bound != float('inf') else float('nan')

        logger.info(f'β={beta}: FN_actual={fn_actual:.6f}, FN_bound={fn_bound:.6f}, tightness={tightness:.4f}')

        results_list.append({
            'beta': beta,
            'fn_actual': fn_actual,
            'fn_bound': fn_bound,
            'tightness_ratio': tightness,
        })

    results = {
        'experiment': 'M5_fn_guarantee',
        'dataset': 'mind',
        'fn_ref': fn_ref,
        'r_max': reward_fn.r_max,
        'results': results_list,
    }
    out_path = os.path.join(RESULTS_DIR, 'M5_fn_guarantee.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
