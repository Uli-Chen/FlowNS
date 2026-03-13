import torch
from flow_matching import VelocityNetwork, FlowMatcher
from models import MatrixFactorization
from trainer import Trainer
import sys

def run_synthetic_test():
    print("=== Starting Synthetic Flow Matching Test ===")
    
    device = 'cpu'
    dim = 64
    batch_size = 32
    n_users = 100
    n_items = 500
    
    print("Initializing Models...")
    mf_model = MatrixFactorization(n_users, n_items, dim).to(device)
    v_net = VelocityNetwork(dim=dim).to(device)
    fm_model = FlowMatcher(v_net).to(device)
    
    print("Creating dummy users...")
    dummy_users = torch.randint(0, n_users, (batch_size,)).to(device)
    user_embs = mf_model.get_embeddings(users=dummy_users)
    
    # 1. Test Base Sampling (alpha=0.0)
    print("\n--- Testing Un-Guided Generation (alpha=0.0) ---")
    gen_embs_base = fm_model.sample(c=user_embs, num_steps=10, alpha=0.0)
    print(f"Generated Vector Shape: {gen_embs_base.shape}")
    
    # Check similarity
    sim_base = (gen_embs_base * user_embs).sum(dim=1).mean().item()
    print(f"Mean Dot Product to User (Base): {sim_base:.4f}")

    # 2. Test Hardness Sampling (alpha=5.0)
    print("\n--- Testing Hardness Guided Generation (alpha=5.0) ---")
    gen_embs_hard = fm_model.sample(c=user_embs, num_steps=10, alpha=5.0)
    
    # Check similarity
    sim_hard = (gen_embs_hard * user_embs).sum(dim=1).mean().item()
    print(f"Mean Dot Product to User (Hard): {sim_hard:.4f}")
    
    if sim_hard > sim_base:
        print(">> SUCCESS! Hardness guidance gradient successfully drifted the negative sample toward the user.")
    else:
        print(">> FAILED! Hardness guidance did not increase similarity.")
        
    # 3. Test Discrete Matching via Trainer Softmax tau
    print("\n--- Testing Discrete Softmax Sampling (tau=0.1) ---")
    
    # Instantiate a dummy trainer to use its method (mocking dataset)
    class DummyDataset:
        n_users = 100
        n_items = 500
        exposures = {}
    trainer = Trainer.__new__(Trainer)
    trainer.mf_model = mf_model
    trainer.fm_model = fm_model
    
    sampled_indices = trainer.sample_discrete_negatives(gen_embs_hard, tau=0.1)
    print(f"Discrete Negatives Shape: {sampled_indices.shape}")
    print(f"First 5 Sampled IDs: {sampled_indices[:5].tolist()}")
    
    print("\nTest completed successfully.")

if __name__ == '__main__':
    run_synthetic_test()
