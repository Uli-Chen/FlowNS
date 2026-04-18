"""Velocity network for Conditional Flow Matching (CondOT).

Predicts the velocity field v(x_t, user_emb, t) used to transport samples
from a source distribution to a target distribution.

Input concatenation: [x_t; user_emb; PE(t)]  where PE(t) is a sinusoidal
positional encoding that maps scalar t to a d-dimensional vector.

Architecture (default):
    Linear(3d, 4d) -> SiLU -> Linear(4d, 4d) -> SiLU -> Linear(4d, d)

Param count: ~35 K for d = 64.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_pe(t: torch.Tensor, d: int) -> torch.Tensor:
    """Sinusoidal positional encoding of scalar time steps.

    Args:
        t: Time values in [0, 1], shape ``[batch]`` or ``[batch, 1]``.
        d: Dimensionality of the output encoding.

    Returns:
        Positional encoding of shape ``[batch, d]``.
    """
    t = t.view(-1, 1).float()  # [batch, 1]

    half_d = d // 2
    # Logarithmically-spaced frequencies akin to the original Transformer PE.
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half_d, device=t.device, dtype=t.dtype) / half_d
    )  # [half_d]

    args = t * freq.unsqueeze(0)  # [batch, half_d]

    # Concatenate sin and cos components.  If d is odd, pad with an extra sin.
    pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [batch, d] when d is even
    if d % 2 == 1:
        pe = torch.cat([pe, torch.sin(args[:, :1])], dim=-1)

    return pe  # [batch, d]


class VelocityNet(nn.Module):
    """Velocity network for Conditional Flow Matching.

    Parameters:
        d:           Embedding / latent dimensionality.
        hidden_mult: Hidden-layer width multiplier (default 4 -> hidden = 4d).
        n_layers:    Number of hidden layers (default 2).
    """

    def __init__(self, d: int, hidden_mult: int = 4, n_layers: int = 2):
        super().__init__()
        self.d = d
        hidden = d * hidden_mult
        input_dim = 3 * d  # [x_t; user_emb; PE(t)]

        layers: list[nn.Module] = []

        # First layer: 3d -> hidden
        layers.append(nn.Linear(input_dim, hidden))
        layers.append(nn.SiLU())

        # Intermediate hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.SiLU())

        # Output projection: hidden -> d
        layers.append(nn.Linear(hidden, d))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier normal initialisation to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        user_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the velocity field.

        Args:
            x_t:      Interpolated point, shape ``[batch, d]``.
            user_emb: User embedding,      shape ``[batch, d]``.
            t:        Scalar time in [0, 1], shape ``[batch]`` or ``[batch, 1]``.

        Returns:
            Velocity prediction of shape ``[batch, d]``.
        """
        t_pe = sinusoidal_pe(t, self.d)  # [batch, d]
        h = torch.cat([x_t, user_emb, t_pe], dim=-1)  # [batch, 3d]
        return self.net(h)  # [batch, d]
