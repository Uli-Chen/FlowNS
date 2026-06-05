"""Pilot M1: Flow pretrain + generation quality check on MIND."""
import sys
import os
import json
import logging
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recbole_utils import setup_recbole, train_recbole, get_embeddings, get_user_positive_items
from src.flow_model import ConditionalFlowModel
from src.sde_sampler import SDESampler
from src.neg_sampling import EmbeddingToItemMapper
from src.custom_metrics import compute_fn_rate

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

    logger.info('=== M1: Flow Pretrain on MIND ===')

    # Step 1: Train LightGCN (or load from M0)
    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', 'mind', config_file_list=config_files, config_dict=config_dict,
    )
    trainer, best_score, _ = train_recbole(config, model, train_data, valid_data)
    logger.info(f'LightGCN valid score: {best_score}')

    # Step 2: Extract embeddings
    user_emb, item_emb = get_embeddings(model)
    user_pos = get_user_positive_items(dataset)
    device = config['device']
    logger.info(f'Embeddings: user={user_emb.shape}, item={item_emb.shape}')

    # Step 3: Pretrain flow model
    flow = ConditionalFlowModel(
        emb_dim=config['embedding_size'], hidden_dim=256, n_layers=3, device=str(device),
    )
    flow.pretrain(user_emb, item_emb, user_pos, epochs=50, lr=1e-4)

    # Step 4: Generate samples and check quality
    sde = SDESampler(flow.velocity_net, n_steps=20, eta=0.5, delta=0.01)
    mapper = EmbeddingToItemMapper(item_emb)

    # Load exposed negatives as ground-truth for FN check
    exposed_neg_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mind', 'mind.exposed_neg')
    test_pos = {}
    inter_matrix = dataset.inter_matrix(form='csr')
    # Use test split positive items for FN measurement
    test_dataset = test_data.dataset if hasattr(test_data, 'dataset') else dataset
    # Simple: use all positive items as potential FN targets
    for uid in range(inter_matrix.shape[0]):
        items = inter_matrix[uid].indices.tolist()
        if items:
            test_pos[uid] = set(items)

    # Sample for 1000 random users
    sample_uids = list(user_pos.keys())[:1000]
    sample_uemb = user_emb[sample_uids]

    flow.velocity_net.eval()
    with torch.no_grad():
        final_emb, _, _ = sde.sample_trajectories(sample_uemb, n_trajectories=1)
        final_emb = final_emb.squeeze(1)
        gen_item_ids = mapper.map_to_items(final_emb)

    fn_rate = compute_fn_rate(gen_item_ids, test_pos, sample_uids)
    logger.info(f'Generated sample FN rate: {fn_rate:.4f}')

    results = {
        'experiment': 'M1_flow_pretrain',
        'dataset': 'mind',
        'flow_pretrain_epochs': 50,
        'fn_rate': fn_rate,
        'n_sample_users': len(sample_uids),
        'decision': 'PASS' if fn_rate < 0.05 else 'INVESTIGATE',
    }
    with open(os.path.join(RESULTS_DIR, 'M1_flow_pretrain.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved. Decision gate: {results["decision"]}')

    # Save flow model
    torch.save(flow.velocity_net.state_dict(), os.path.join(RESULTS_DIR, 'flow_pretrained.pt'))
    torch.save(flow.ref_state_dict, os.path.join(RESULTS_DIR, 'flow_ref.pt'))
    logger.info('Flow model saved.')


if __name__ == '__main__':
    main()
