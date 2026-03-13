import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(MatrixFactorization, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # User and Item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self._init_weights()

    def _init_weights(self):
        # Xavier/Normal initialization is common, let's use normal.
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, users, items):
        """
        Calculates the dot product score between users and items
        users: tensor of user indices
        items: tensor of item indices
        Returns: scores tensor
        """
        user_emb = self.user_embedding(users)    # [Batch, Dim]
        item_emb = self.item_embedding(items)    # [Batch, Dim]
        
        # Element-wise multiplication followed by sum over dimension 1 (dot product)
        scores = (user_emb * item_emb).sum(dim=1)
        return scores

    def get_embeddings(self, users=None, items=None):
        if users is not None:
            return self.user_embedding(users)
        elif items is not None:
            return self.item_embedding(items)
        return self.user_embedding.weight, self.item_embedding.weight

def bpr_loss(pos_scores, neg_scores):
    """
    Bayesian Personalized Ranking (BPR) Loss
    pos_scores: scores of positive items [Batch]
    neg_scores: scores of negative items [Batch]
    """
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

if __name__ == '__main__':
    # Simple test for models
    model = MatrixFactorization(num_users=100, num_items=100, embedding_dim=32)
    dummy_users = torch.randint(0, 100, (10,))
    dummy_pos = torch.randint(0, 100, (10,))
    dummy_neg = torch.randint(0, 100, (10,))
    
    pos_scores = model(dummy_users, dummy_pos)
    neg_scores = model(dummy_users, dummy_neg)
    
    loss = bpr_loss(pos_scores, neg_scores)
    print("Dummy BPR Loss:", loss.item())
