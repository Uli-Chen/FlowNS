from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)

    def score_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_embedding(users) * self.item_embedding(items)).sum(dim=-1)

    def score_all(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_embedding(users) @ self.item_embedding.weight.t()

    def bpr_loss(
        self,
        users: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
    ) -> torch.Tensor:
        positive_scores = self.score_pairs(users, positive_items)
        negative_scores = self.score_pairs(users, negative_items)
        return -F.logsigmoid(positive_scores - negative_scores).mean()


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = sequences.shape
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.item_embedding(sequences) + self.position_embedding(positions)
        hidden = self.dropout(self.layer_norm(hidden))
        attention_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=sequences.device),
            diagonal=1,
        )
        padding_mask = sequences.eq(0)
        hidden = self.encoder(hidden, mask=attention_mask, src_key_padding_mask=padding_mask)
        lengths = sequences.ne(0).sum(dim=1).clamp_min(1) - 1
        return hidden[torch.arange(batch_size, device=sequences.device), lengths]

    def score_items(self, sequences: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_repr = self.encode(sequences)
        return (user_repr * self.item_embedding(items)).sum(dim=-1)

    def score_all(self, sequences: torch.Tensor) -> torch.Tensor:
        user_repr = self.encode(sequences)
        return user_repr @ self.item_embedding.weight[1:].t()

    def bpr_loss(
        self,
        sequences: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
    ) -> torch.Tensor:
        positive_scores = self.score_items(sequences, positive_items)
        negative_scores = self.score_items(sequences, negative_items)
        return -F.logsigmoid(positive_scores - negative_scores).mean()
