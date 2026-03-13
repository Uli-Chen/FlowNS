import torch
import torch.nn as nn

class VelocityNetwork(nn.Module):
    """
    Predicts the velocity field for Flow Matching.
    Input:
    - t: time step [Batch, 1]
    - x_t: intermediate state [Batch, Dim]
    - c: context/condition (e.g., user embedding) [Batch, Dim]
    """
    def __init__(self, dim, hidden_dims=[256, 256, 256]):
        super(VelocityNetwork, self).__init__()
        
        layers = []
        # Input dim is dim (x_t) + dim (c) + 1 (t)
        in_dim = dim * 2 + 1
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.SiLU()) # Swish activation is common for score/flow networks
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x_t, c):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        # Handle broadcasting if t is a scalar but x_t is batched
        if t.size(0) != x_t.size(0):
            t = t.expand(x_t.size(0), -1)
            
        x = torch.cat([t, x_t, c], dim=-1)
        return self.net(x)

class FlowMatcher(nn.Module):
    """
    Handles Flow Matching training and ODE sampling.
    """
    def __init__(self, velocity_network):
        super(FlowMatcher, self).__init__()
        self.v_net = velocity_network

    def compute_loss(self, x_1, c):
        """
        Computes the Conditional Flow Matching (CFM) loss.
        x_1: Target items from the exposure distribution [Batch, Dim]
        c: User condition [Batch, Dim]
        """
        batch_size = x_1.size(0)
        
        # 1. Sample t ~ U(0, 1)
        t = torch.rand(batch_size, 1, device=x_1.device)
        
        # 2. Sample x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)
        
        # 3. Construct x_t = t * x_1 + (1 - t) * x_0 (OT-Flow path)
        x_t = t * x_1 + (1 - t) * x_0
        
        # 4. Target velocity is x_1 - x_0
        target_v = x_1 - x_0
        
        # 5. Predict velocity
        pred_v = self.v_net(t, x_t, c)
        
        # 6. MSE Loss
        loss = torch.nn.functional.mse_loss(pred_v, target_v)
        return loss

    @torch.no_grad()
    def sample(self, c, num_steps=50, alpha=0.0):
        """
        Euler method to sample from the learned flow.
        c: Context/User embedding [Batch, Dim]
        num_steps: Number of integration steps
        alpha: Guidance strength for hardness (0.0 means unguided)
        Returns: Generated negative samples [Batch, Dim]
        """
        batch_size = c.size(0)
        dim = c.size(1)
        device = c.device
        
        # Start from base distribution x_0 ~ N(0, I)
        x_0 = torch.randn(batch_size, dim, device=device)
        
        if alpha > 0.0:
            with torch.enable_grad():
                x_0.requires_grad_(True)
                
                # t=0 velocity
                t_0 = torch.zeros((batch_size, 1), device=device)
                v_0 = self.v_net(t_0, x_0, c)
                
                # Approximate x_1
                x_1_approx = x_0 + v_0
                
                # Compute score: dot product similarity to condition/user
                score = (x_1_approx * c).sum()
                
                # Gradient of score with respect to x_0
                grad = torch.autograd.grad(score, x_0)[0]
                
            # Update initialization
            x_t = x_0.detach() + alpha * grad
        else:
            x_t = x_0
        
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = torch.full((batch_size, 1), step * dt, device=device)
            v = self.v_net(t, x_t, c)
            x_t = x_t + v * dt
            
        return x_t

if __name__ == '__main__':
    # Simple test
    dim = 64
    batch_size = 32
    v_net = VelocityNetwork(dim=dim)
    matcher = FlowMatcher(v_net)
    
    dummy_x1 = torch.randn(batch_size, dim)
    dummy_c = torch.randn(batch_size, dim)
    
    loss = matcher.compute_loss(dummy_x1, dummy_c)
    print("Dummy Flow Matching Loss:", loss.item())
    
    generated = matcher.sample(dummy_c, num_steps=20)
    print("Generated shape:", generated.shape)
