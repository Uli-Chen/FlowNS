import torch


class BoundaryAwareReward:
    """R = W^a · (1-W)^γ with OPAUC-motivated win rate."""

    def __init__(self, a=1.0, gamma=1.0):
        self.a = a
        self.gamma = gamma

    def win_rate(self, user_emb, gen_emb, pos_item_embs, pos_mask=None):
        """Compute win rate W(x, u) = (1/|I_u+|) Σ σ(s(u,x) - s(u,i)).

        Args:
            user_emb: (B, d)
            gen_emb: (B, d) generated negative embeddings
            pos_item_embs: (B, K, d) positive item embeddings for each user
            pos_mask: optional (B, K) bool marking real (non-padded) positives.
                Without it, zero-padded slots contribute σ(s_gen) ≈ 0.5 to the
                mean and bias W toward W* for users with fewer than K positives.
        Returns:
            W: (B,) win rates in (0, 1)
        """
        score_gen = (user_emb * gen_emb).sum(dim=-1, keepdim=True)  # (B, 1)
        score_pos = (user_emb.unsqueeze(1) * pos_item_embs).sum(dim=-1)  # (B, K)
        wins = torch.sigmoid(score_gen - score_pos)  # (B, K)
        if pos_mask is None:
            return wins.mean(dim=-1)
        mask = pos_mask.to(wins.dtype)
        return (wins * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)

    def compute_reward(self, W):
        """R = W^a · (1-W)^γ.

        Args:
            W: (B,) win rates
        Returns:
            R: (B,) rewards
        """
        W_clamped = W.clamp(1e-7, 1 - 1e-7)
        return W_clamped.pow(self.a) * (1 - W_clamped).pow(self.gamma)

    @property
    def r_max(self):
        """R_max = a^a · γ^γ / (a+γ)^(a+γ)."""
        a, g = self.a, self.gamma
        return (a ** a * g ** g) / (a + g) ** (a + g)

    @property
    def optimal_w(self):
        """W* = a / (a + γ)."""
        return self.a / (self.a + self.gamma)

    def __repr__(self):
        return (f'BoundaryAwareReward(a={self.a}, γ={self.gamma}, '
                f'W*={self.optimal_w:.3f}, R_max={self.r_max:.6f})')
