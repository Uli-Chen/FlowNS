import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from dataset import MINDDataset, ExposureDataset
from models import MatrixFactorization, bpr_loss
from flow_matching import VelocityNetwork, FlowMatcher
from utils import evaluate_model

class Trainer:
    def __init__(self, data_dir, device='cuda' if torch.cuda.is_available() else 'cpu',
                 embedding_dim=64, batch_size=1024):
        self.device = device
        self.batch_size = batch_size
        self.dim = embedding_dim
        
        self.dataset = MINDDataset(data_dir)
        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items
        
        # Initialize Models
        self.mf_model = MatrixFactorization(self.n_users, self.n_items, self.dim).to(self.device)
        self.mf_optimizer = optim.Adam(self.mf_model.parameters(), lr=0.001)
        
        self.v_net = VelocityNetwork(dim=self.dim).to(self.device)
        self.fm_model = FlowMatcher(self.v_net).to(self.device)
        self.fm_optimizer = optim.Adam(self.fm_model.parameters(), lr=0.001)

    def train_mf_stage1(self, epochs=5):
        print("=== Stage 1: Pretrain MF with Random Negative Sampling ===")
        train_ds = self.dataset.get_train_dataset(num_negatives=1)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.mf_model.train()
            total_loss = 0
            start_t = time.time()
            for batch in train_dl:
                users = batch['user'].to(self.device)
                pos_items = batch['pos_item'].to(self.device)
                neg_items = batch['neg_item'].to(self.device)
                
                pos_scores = self.mf_model(users, pos_items)
                neg_scores = self.mf_model(users, neg_items)
                
                loss = bpr_loss(pos_scores, neg_scores)
                
                self.mf_optimizer.zero_grad()
                loss.backward()
                self.mf_optimizer.step()
                
                total_loss += loss.item()
                
            elapsed = time.time() - start_t
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_dl):.4f} | Time: {elapsed:.2f}s")
            
            # Evaluate at end of stage
            if (epoch + 1) % epochs == 0 or (epoch + 1) % 5 == 0:
                metrics = evaluate_model(self.mf_model, self.dataset, topk=[10, 20], device=self.device)
                print(f"--- Eval --- NDCG@10: {metrics[10]['ndcg']:.4f}  Recall@10: {metrics[10]['recall']:.4f}")
                print(f"--- Eval --- NDCG@20: {metrics[20]['ndcg']:.4f}  Recall@20: {metrics[20]['recall']:.4f}")
            
    def train_fm_stage2(self, epochs=5):
        print("=== Stage 2: Train Flow Matching Generator ===")
        exp_ds = ExposureDataset(self.dataset.exposures)
        exp_dl = DataLoader(exp_ds, batch_size=self.batch_size, shuffle=True)
        
        # Freeze MF embeddings
        self.mf_model.eval()
        for param in self.mf_model.parameters():
            param.requires_grad = False
            
        for epoch in range(epochs):
            self.fm_model.train()
            total_loss = 0
            start_t = time.time()
            for batch in exp_dl:
                users = batch['user'].to(self.device)
                items = batch['item'].to(self.device)
                
                with torch.no_grad():
                    user_embs = self.mf_model.get_embeddings(users=users)
                    item_embs = self.mf_model.get_embeddings(items=items)
                
                # Flow matching loss
                loss = self.fm_model.compute_loss(x_1=item_embs, c=user_embs)
                
                self.fm_optimizer.zero_grad()
                loss.backward()
                self.fm_optimizer.step()
                
                total_loss += loss.item()
                
            elapsed = time.time() - start_t
            print(f"Epoch {epoch+1}/{epochs} | CFM Loss: {total_loss/len(exp_dl):.4f} | Time: {elapsed:.2f}s")
            
        # Unfreeze MF
        for param in self.mf_model.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def sample_discrete_negatives(self, generated_embs, tau=1.0):
        """
        Maps continuous generated embeddings to discrete item IDs
        via softmax probability distribution.
        """
        all_item_embs = self.mf_model.get_embeddings()[-1] # [N_items, Dim]
        
        # Compute similarities/logits: [Batch, N_items]
        # generated_embs: [Batch, Dim]
        logits = torch.matmul(generated_embs, all_item_embs.T)
        
        # Softmax distribution with temperature
        probs = torch.softmax(logits / tau, dim=-1)
        
        # Sample discrete indices
        sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1) # [Batch]
        return sampled_indices

    def finetune_mf_stage3(self, epochs=5, alpha_max=1.0, tau_start=1.0, tau_end=0.1):
        print("=== Stage 3: Finetune MF with Flow Matching Negatives ===")
        train_ds = self.dataset.get_train_dataset(num_negatives=0) # Only get positives
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        self.fm_model.eval()
        
        for epoch in range(epochs):
            self.mf_model.train()
            total_loss = 0
            start_t = time.time()
            
            # Hardness Curriculum Scheduling
            t_ratio = epoch / max(1, epochs - 1) # 0.0 to 1.0
            current_alpha = alpha_max * t_ratio
            current_tau = tau_start - (tau_start - tau_end) * t_ratio
            
            for batch in train_dl:
                users = batch['user'].to(self.device)
                pos_items = batch['pos_item'].to(self.device)
                
                # Generate Negatives
                with torch.no_grad():
                    user_embs = self.mf_model.get_embeddings(users=users)
                    gen_embs = self.fm_model.sample(c=user_embs, num_steps=10, alpha=current_alpha) 
                    neg_items = self.sample_discrete_negatives(gen_embs, tau=current_tau)
                
                pos_scores = self.mf_model(users, pos_items)
                neg_scores = self.mf_model(users, neg_items)
                
                loss = bpr_loss(pos_scores, neg_scores)
                
                self.mf_optimizer.zero_grad()
                loss.backward()
                self.mf_optimizer.step()
                
                total_loss += loss.item()
                
            elapsed = time.time() - start_t
            print(f"Epoch {epoch+1}/{epochs} | Finetune Loss: {total_loss/len(train_dl):.4f} | Time: {elapsed:.2f}s | alpha: {current_alpha:.2f} | tau: {current_tau:.2f}")

            # Evaluate at end of stage
            if (epoch + 1) % epochs == 0 or (epoch + 1) % 5 == 0:
                metrics = evaluate_model(self.mf_model, self.dataset, topk=[10, 20], device=self.device)
                print(f"--- Eval --- NDCG@10: {metrics[10]['ndcg']:.4f}  Recall@10: {metrics[10]['recall']:.4f}")
                print(f"--- Eval --- NDCG@20: {metrics[20]['ndcg']:.4f}  Recall@20: {metrics[20]['recall']:.4f}")
