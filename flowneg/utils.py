"""Utility functions for FlowNeg: ODE solver, positional encoding, temperature schedule."""

import math
import torch


def midpoint_solver(v_theta, x_0, user_emb, n_steps=4):
    """N-step midpoint ODE solver for flow matching.

    Integrates dx/dt = v_theta(x, user_emb, t) from t=0 to t=1
    using the explicit midpoint (modified Euler) method.

    Args:
        v_theta: Callable (x, user_emb, t) -> velocity, where
                 x: [batch, d], user_emb: [batch, d], t: [batch, 1] or scalar.
        x_0: Initial condition, shape [batch, d].
        user_emb: User embedding, shape [batch, d].
        n_steps: Number of integration steps (default 4).

    Returns:
        x_1: Solution at t=1, shape [batch, d].
    """
    dt = 1.0 / n_steps
    x = x_0.clone()

    for i in range(n_steps):
        t_start = i * dt
        t_mid = t_start + dt / 2.0

        # Broadcast time to [batch, 1]
        batch_size = x.shape[0]
        t_start_tensor = torch.full((batch_size, 1), t_start, device=x.device, dtype=x.dtype)
        t_mid_tensor = torch.full((batch_size, 1), t_mid, device=x.device, dtype=x.dtype)

        # Midpoint method: evaluate at start, step to midpoint, evaluate there
        k1 = v_theta(x, user_emb, t_start_tensor)
        x_mid = x + (dt / 2.0) * k1
        k2 = v_theta(x_mid, user_emb, t_mid_tensor)
        x = x + dt * k2

    return x


def sinusoidal_pe(t, d):
    """Sinusoidal positional encoding for scalar time values.

    Encodes each scalar time value into a d-dimensional vector using
    sin/cos at logarithmically spaced frequencies, following the
    Transformer positional encoding scheme.

    Args:
        t: Time values, shape [batch] or [batch, 1].
        d: Output embedding dimension (must be even).

    Returns:
        Positional encoding, shape [batch, d].
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)  # [batch, 1]

    assert d % 2 == 0, f"Embedding dimension d must be even, got {d}"

    # Log-spaced frequencies: 1 / 10000^(2i/d) for i = 0, ..., d/2 - 1
    half_d = d // 2
    freq_exponents = torch.arange(half_d, device=t.device, dtype=t.dtype) / half_d
    inv_freq = torch.exp(-math.log(10000.0) * freq_exponents)  # [d/2]

    # Outer product: [batch, 1] * [1, d/2] -> [batch, d/2]
    angles = t * inv_freq.unsqueeze(0)

    # Interleave sin and cos: [batch, d]
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return pe


def tau_schedule(epoch, max_epochs, schedule='cosine', tau_start=2.0, tau_end=0.1):
    """Temperature schedule for controlling sampling sharpness over training.

    Args:
        epoch: Current epoch (0-indexed).
        max_epochs: Total number of epochs.
        schedule: One of 'cosine', 'linear', 'constant'.
        tau_start: Initial temperature.
        tau_end: Final temperature (used by cosine and linear).

    Returns:
        tau: Temperature value (float).
    """
    if schedule == 'constant':
        return tau_start

    # Clamp progress to [0, 1]
    progress = min(max(epoch / max(max_epochs - 1, 1), 0.0), 1.0)

    if schedule == 'cosine':
        # Cosine annealing: smoothly interpolates from tau_start to tau_end
        tau = tau_end + 0.5 * (tau_start - tau_end) * (1.0 + math.cos(math.pi * progress))
        return tau

    if schedule == 'linear':
        tau = tau_start + (tau_end - tau_start) * progress
        return tau

    raise ValueError(f"Unknown schedule '{schedule}'. Choose from 'cosine', 'linear', 'constant'.")
