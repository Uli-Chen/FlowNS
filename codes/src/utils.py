import torch
import numpy as np

def compute_metrics(rank_list, ground_truth, k_list=[10, 20]):
    """
    Computes Recall and NDCG at specified K values for a single user's rank list.
    rank_list: [N] sorted tensor of item IDs (highest score first)
    ground_truth: list or set of true positive item IDs for this user
    k_list: list of K values to evaluate
    Returns: dict mapping k -> {'recall': val, 'ndcg': val}
    """
    metrics = {k: {'recall': 0.0, 'ndcg': 0.0} for k in k_list}
    if len(ground_truth) == 0:
        return metrics
        
    for k in k_list:
        topk_items = rank_list[:k].tolist()
        hits = 0.0
        dcg = 0.0
        for i, item in enumerate(topk_items):
            if item in ground_truth:
                hits += 1.0
                dcg += 1.0 / np.log2(i + 2) # i is 0-indexed, so rank is i+1, log2(rank+1) -> log2(i+2)
        
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k))])
        
        metrics[k]['recall'] = hits / len(ground_truth)
        metrics[k]['ndcg'] = dcg / idcg if idcg > 0 else 0.0
        
    return metrics

@torch.no_grad()
def evaluate_model(model, dataset, topk=[10, 20], device='cuda'):
    """
    Performs full-ranking evaluation on the test dataset.
    model: Recommendation model (e.g. MatrixFactorization)
    dataset: MINDDataset instance
    """
    model.eval()
    test_interactions, train_interactions = dataset.get_test_dataset()
    all_item_ids = torch.arange(dataset.n_items, device=device)
    
    overall_metrics = {k: {'recall': [], 'ndcg': []} for k in topk}
    
    # We evaluate sequentially per user to avoid OOM for huge datasets,
    # but process in batches of users for speed
    users = list(test_interactions.keys())
    batch_size = 256
    
    for start_idx in range(0, len(users), batch_size):
        end_idx = min(start_idx + batch_size, len(users))
        batch_users = users[start_idx:end_idx]
        
        # [Batch, N_items] user-item scores
        batch_users_tensor = torch.tensor(batch_users, dtype=torch.long, device=device).unsqueeze(1) # [Batch, 1]
        batch_items_tensor = all_item_ids.unsqueeze(0).expand(len(batch_users), -1) # [Batch, N_items]
        
        # To compute scores across all items efficiently, we can use model.get_embeddings
        user_embs = model.get_embeddings(users=torch.tensor(batch_users, dtype=torch.long, device=device)) # [Batch, Dim]
        item_embs = model.get_embeddings()[-1] # [N_items, Dim]
        
        # Scores: [Batch, N_items]
        scores = torch.matmul(user_embs, item_embs.T)
        
        for i, u in enumerate(batch_users):
            u_ground_truth = test_interactions[u]
            if len(u_ground_truth) == 0:
                continue
                
            u_train_items = train_interactions[u]
            
            # Mask out training items
            scores[i, u_train_items] = -float('inf')
            
            # Sort items by score
            _, rank_list = torch.sort(scores[i], descending=True)
            
            # Compute metrics
            u_metrics = compute_metrics(rank_list, u_ground_truth, k_list=topk)
            
            for k in topk:
                overall_metrics[k]['recall'].append(u_metrics[k]['recall'])
                overall_metrics[k]['ndcg'].append(u_metrics[k]['ndcg'])
                
    # Average metrics
    avg_metrics = {}
    for k in topk:
        avg_metrics[k] = {
            'recall': np.mean(overall_metrics[k]['recall']) if overall_metrics[k]['recall'] else 0.0,
            'ndcg': np.mean(overall_metrics[k]['ndcg']) if overall_metrics[k]['ndcg'] else 0.0
        }
        
    return avg_metrics

if __name__ == '__main__':
    # Test script usage
    pass
