import torch
import torch.nn as nn
import torch.nn.functional as F


def bpr_loss(pos_scores, neg_scores):
    """
    Bayesian Personalized Ranking (BPR) loss.
    """
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def get_user_embedding(self, users):
        return self.user_embedding(users)

    def get_item_embedding(self, items):
        return self.item_embedding(items)

    def forward(self, users, items):
        """
        Dot-product score for (user, item) pairs.
        """
        user_emb = self.get_user_embedding(users)
        item_emb = self.get_item_embedding(items)
        return (user_emb * item_emb).sum(dim=1)

    def predict(self, users, items):
        return self.forward(users, items)

    def full_sort_predict(self, users):
        user_emb = self.get_user_embedding(users)
        all_item_emb = self.item_embedding.weight
        return torch.matmul(user_emb, all_item_emb.transpose(0, 1))

    def calculate_loss(self, batch):
        users = batch["user"]
        pos_items = batch["pos_item"]
        neg_items = batch["neg_item"]
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        return bpr_loss(pos_scores, neg_scores)

    def get_embeddings(self, users=None, items=None):
        if users is not None:
            return self.user_embedding(users)
        if items is not None:
            return self.item_embedding(items)
        return self.user_embedding.weight, self.item_embedding.weight


class MultiVAE(nn.Module):
    """
    MultiVAE for implicit-feedback recommendation.
    """

    def __init__(
        self,
        num_users,
        num_items,
        train_interactions=None,
        hidden_dims=None,
        latent_dim=64,
        dropout=0.5,
        anneal_cap=0.2,
        total_anneal_steps=200000,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.train_interactions = train_interactions or {}

        self.hidden_dims = hidden_dims or [600, 200]
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.anneal_cap = anneal_cap
        self.total_anneal_steps = total_anneal_steps
        self.update_count = 0

        encoder_dims = [num_items] + self.hidden_dims + [latent_dim * 2]
        decoder_dims = [latent_dim] + self.hidden_dims[::-1] + [num_items]

        self.encoder = self._build_mlp(encoder_dims)
        self.decoder = self._build_mlp(decoder_dims)
        self.reset_parameters()

    def _build_mlp(self, dims):
        layers = []
        for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < len(dims) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_train_interactions(self, train_interactions):
        self.train_interactions = train_interactions

    def _reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std) * 0.01
        return mu + epsilon * std

    def build_rating_matrix(self, users, device=None):
        if device is None:
            device = users.device
        rating_matrix = torch.zeros((users.size(0), self.num_items), device=device)
        for row_idx, uid in enumerate(users.tolist()):
            items = self.train_interactions.get(uid, [])
            if items:
                rating_matrix[row_idx, items] = 1.0
        return rating_matrix

    def forward(self, rating_matrix):
        x = F.normalize(rating_matrix, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.encoder(x)
        mu = h[:, : self.latent_dim]
        logvar = h[:, self.latent_dim :]
        z = self._reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar

    def _get_anneal(self):
        if self.total_anneal_steps <= 0:
            return self.anneal_cap
        return min(self.anneal_cap, self.update_count / self.total_anneal_steps)

    def calculate_loss(self, users_or_batch):
        if isinstance(users_or_batch, dict):
            users = users_or_batch["user"]
        else:
            users = users_or_batch
        rating_matrix = self.build_rating_matrix(users)

        self.update_count += 1
        anneal = self._get_anneal()

        logits, mu, logvar = self.forward(rating_matrix)
        ce_loss = -(F.log_softmax(logits, dim=1) * rating_matrix).sum(dim=1).mean()
        kl_raw = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return ce_loss + anneal * kl_raw

    @torch.no_grad()
    def full_sort_predict(self, users):
        rating_matrix = self.build_rating_matrix(users)
        logits, _, _ = self.forward(rating_matrix)
        return logits

    @torch.no_grad()
    def predict(self, users, items):
        logits = self.full_sort_predict(users)
        row_ids = torch.arange(items.size(0), device=items.device)
        return logits[row_ids, items]

if __name__ == '__main__':
    model = MatrixFactorization(num_users=100, num_items=100, embedding_dim=32)
    users = torch.randint(0, 100, (10,))
    pos = torch.randint(0, 100, (10,))
    neg = torch.randint(0, 100, (10,))
    loss = model.calculate_loss({"user": users, "pos_item": pos, "neg_item": neg})
    print("Dummy BPR loss:", loss.item())
