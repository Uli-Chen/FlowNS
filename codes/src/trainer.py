import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import copy

try:
    import wandb
except ImportError:
    wandb = None

from dataset import MINDDataset, ML100KDataset, ExposureDataset
from models import MatrixFactorization, bpr_loss
from flow_matching import VelocityNetwork, FlowMatcher
from utils import evaluate_model


class Trainer:
    def __init__(self, data_dir, device='cuda' if torch.cuda.is_available() else 'cpu',
                 embedding_dim=64, batch_size=1024, val_dir=None, eval_every=5,
                 dataset_type='mind', stage1_only=False, patience=10,
                 wandb_project='FlowNS', wandb_entity=None, wandb_name=None,
                 wandb_mode='offline', run_config=None):
        self.device = device
        self.batch_size = batch_size
        self.dim = embedding_dim
        self.eval_every = eval_every
        self.patience = patience
        
        # Load dataset based on type
        if dataset_type == 'ml100k':
            self.dataset = ML100KDataset(data_dir)
        else:
            self.dataset = MINDDataset(data_dir)
        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items
        
        # Load validation data if provided (MIND only)
        self.has_val = False
        if val_dir is not None and dataset_type == 'mind':
            self.dataset.load_validation_data(val_dir)
            self.has_val = True
        
        # Initialize Models
        self.mf_model = MatrixFactorization(self.n_users, self.n_items, self.dim).to(self.device)
        self.mf_optimizer = optim.Adam(self.mf_model.parameters(), lr=0.001)
        
        if not stage1_only:
            self.v_net = VelocityNetwork(dim=self.dim).to(self.device)
            self.fm_model = FlowMatcher(self.v_net).to(self.device)
            self.fm_optimizer = optim.Adam(self.fm_model.parameters(), lr=0.001)

        self.wandb_enabled = False
        self.wandb_run = None
        self._init_wandb(
            project=wandb_project,
            entity=wandb_entity,
            run_name=wandb_name,
            mode=wandb_mode,
            run_config=run_config
        )

    def _init_wandb(self, project, entity, run_name, mode, run_config):
        if mode == 'disabled':
            return
        if wandb is None:
            print(">>> wandb is not installed, fallback to no external logging.")
            return

        init_kwargs = {
            'project': project,
            'mode': mode,
            'config': run_config or {},
        }
        if entity:
            init_kwargs['entity'] = entity
        if run_name:
            init_kwargs['name'] = run_name

        try:
            self.wandb_run = wandb.init(**init_kwargs)
            self.wandb_enabled = self.wandb_run is not None
        except Exception as exc:
            print(f">>> Failed to init wandb ({exc}), fallback to no external logging.")
            self.wandb_enabled = False
            self.wandb_run = None

    def _log_wandb(self, payload, step):
        if self.wandb_enabled and self.wandb_run is not None:
            wandb.log(payload, step=step)

    def _evaluate_test(self, topk=[10, 20]):
        """Evaluate on test set and return metrics dict."""
        test_interactions, train_interactions = self.dataset.get_test_dataset()
        return evaluate_model(self.mf_model, test_interactions, train_interactions,
                              self.n_items, topk=topk, device=self.device)

    def _evaluate_val(self, topk=[10, 20]):
        """Evaluate on validation set and return metrics dict."""
        if not self.has_val:
            return None
        val_interactions, train_interactions = self.dataset.get_val_dataset()
        return evaluate_model(self.mf_model, val_interactions, train_interactions,
                              self.n_items, topk=topk, device=self.device)

    def _log_metrics(self, metrics, prefix, global_step, topk=[10, 20]):
        """Log metrics and print to console."""
        log_payload = {}
        for k in topk:
            log_payload[f'{prefix}/NDCG@{k}'] = metrics[k]['ndcg']
            log_payload[f'{prefix}/Recall@{k}'] = metrics[k]['recall']
        self._log_wandb(log_payload, step=global_step)
        print(f"  [{prefix}] NDCG@10: {metrics[10]['ndcg']:.4f}  Recall@10: {metrics[10]['recall']:.4f}"
              f"  |  NDCG@20: {metrics[20]['ndcg']:.4f}  Recall@20: {metrics[20]['recall']:.4f}")

    def _print_final_test(self, stage_name, best_epoch, best_state):
        """Load best model, evaluate on test set, and print results."""
        self.mf_model.load_state_dict(best_state)
        print(f"\n>>> Best model from epoch {best_epoch} restored. Running final test evaluation...")
        metrics = self._evaluate_test()
        self._log_metrics(metrics, f'{stage_name}/Final_Test', best_epoch)
        
        if self.has_val:
            val_metrics = self._evaluate_val()
            self._log_metrics(val_metrics, f'{stage_name}/Final_Val', best_epoch)

    def train_mf_stage1(self, epochs=5):
        print("=== Stage 1: Pretrain MF with Random Negative Sampling ===")
        train_ds = self.dataset.get_train_dataset(num_negatives=1)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        best_ndcg = 0.0
        best_epoch = 0
        best_state = None
        no_improve_count = 0
        
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
                
            avg_loss = total_loss / len(train_dl)
            elapsed = time.time() - start_t
            self._log_wandb({'S1/Loss': avg_loss}, step=epoch + 1)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
            
            # Periodic evaluation for early stopping
            if (epoch + 1) % self.eval_every == 0:
                metrics = self._evaluate_test()
                self._log_metrics(metrics, 'S1/Test', epoch + 1)
                
                if self.has_val:
                    val_metrics = self._evaluate_val()
                    self._log_metrics(val_metrics, 'S1/Val', epoch + 1)
                
                # Early stopping check (based on test NDCG@20)
                current_ndcg = metrics[20]['ndcg']
                if current_ndcg > best_ndcg:
                    best_ndcg = current_ndcg
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(self.mf_model.state_dict())
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count >= self.patience:
                    print(f">>> Early stopping at epoch {epoch+1}. No improvement for {self.patience} eval cycles.")
                    break
        
        # Final test with best model
        if best_state is not None:
            self._print_final_test('S1', best_epoch, best_state)
        print(f">>> Stage 1 finished. Best epoch: {best_epoch} | Best NDCG@20: {best_ndcg:.4f}\n")
            
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
                
            avg_loss = total_loss / len(exp_dl)
            elapsed = time.time() - start_t
            self._log_wandb({'S2/CFM_Loss': avg_loss}, step=epoch + 1)
            print(f"Epoch {epoch+1}/{epochs} | CFM Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
            
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
        
        best_ndcg = 0.0
        best_epoch = 0
        best_state = None
        no_improve_count = 0
        
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
                
            avg_loss = total_loss / len(train_dl)
            elapsed = time.time() - start_t
            self._log_wandb({
                'S3/Loss': avg_loss,
                'S3/alpha': current_alpha,
                'S3/tau': current_tau
            }, step=epoch + 1)
            print(f"Epoch {epoch+1}/{epochs} | Finetune Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s | alpha: {current_alpha:.2f} | tau: {current_tau:.2f}")

            # Periodic evaluation for early stopping
            if (epoch + 1) % self.eval_every == 0:
                metrics = self._evaluate_test()
                self._log_metrics(metrics, 'S3/Test', epoch + 1)
                
                if self.has_val:
                    val_metrics = self._evaluate_val()
                    self._log_metrics(val_metrics, 'S3/Val', epoch + 1)
                
                # Early stopping check
                current_ndcg = metrics[20]['ndcg']
                if current_ndcg > best_ndcg:
                    best_ndcg = current_ndcg
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(self.mf_model.state_dict())
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count >= self.patience:
                    print(f">>> Early stopping at epoch {epoch+1}. No improvement for {self.patience} eval cycles.")
                    break
        
        # Final test with best model
        if best_state is not None:
            self._print_final_test('S3', best_epoch, best_state)
        print(f">>> Stage 3 finished. Best epoch: {best_epoch} | Best NDCG@20: {best_ndcg:.4f}\n")

    def close(self):
        """Close logger resources."""
        if self.wandb_enabled and self.wandb_run is not None:
            wandb.finish()
