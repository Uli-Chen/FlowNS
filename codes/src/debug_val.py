"""Diagnostic: directly run evaluate_model on val data and check for hits."""
import torch
from dataset import MINDDataset
from models import MatrixFactorization
from utils import evaluate_model
import numpy as np

device = 'cpu'

# Load data
ds = MINDDataset('data/MINDsmall_train')
ds.load_validation_data('data/MINDsmall_dev/MINDsmall_dev')

# Create model and do a few training steps to simulate
model = MatrixFactorization(ds.n_users, ds.n_items, 64).to(device)

# Get val data
val_inter, train_inter = ds.get_val_dataset()
test_inter, _ = ds.get_test_dataset()

print(f"Val users: {len(val_inter)}, Val pairs: {sum(len(v) for v in val_inter.values())}")
print(f"Test users: {len(test_inter)}, Test pairs: {sum(len(v) for v in test_inter.values())}")

# Manually check: for a few val users, what rank does their ground truth item get?
model.eval()
all_item_embs = model.get_embeddings()[-1]  # [N_items, Dim]

sample_users = list(val_inter.keys())[:10]
for u in sample_users:
    gt_items = val_inter[u]
    train_items = train_inter.get(u, [])
    
    user_emb = model.get_embeddings(users=torch.tensor([u], dtype=torch.long))
    scores = torch.matmul(user_emb, all_item_embs.T).squeeze(0)
    scores[train_items] = -float('inf')
    
    _, rank_list = torch.sort(scores, descending=True)
    rank_list = rank_list.tolist()
    
    for gt in gt_items:
        rank = rank_list.index(gt) + 1 if gt in rank_list else -1
        print(f"  User {u}: GT item {gt} -> rank {rank}/{ds.n_items} (train items masked: {len(train_items)})")

# Run full evaluation with high precision
metrics = evaluate_model(model, val_inter, train_inter, ds.n_items, topk=[10, 20, 100], device=device)
print(f"\n[VAL] Recall@10:  {metrics[10]['recall']:.8f}")
print(f"[VAL] Recall@20:  {metrics[20]['recall']:.8f}")  
print(f"[VAL] Recall@100: {metrics[100]['recall']:.8f}")
print(f"[VAL] NDCG@10:    {metrics[10]['ndcg']:.8f}")
print(f"[VAL] NDCG@20:    {metrics[20]['ndcg']:.8f}")
print(f"[VAL] NDCG@100:   {metrics[100]['ndcg']:.8f}")
