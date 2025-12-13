import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LLMConfig


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    This is a gated MLP that uses the Swish activation function.
    It computes: output = W_down · (Swish(W_gate · x) ⊙ W_up · x)
    
    The gating mechanism allows the network to learn which features
    to amplify or suppress, leading to better performance than
    standard ReLU MLPs.
    
    Architecture:
        x ──┬──→ W_gate ──→ Swish ──┐
            │                       ⊙ ──→ W_down ──→ output
            └──→ W_up ─────────────┘
    """
    
    def __init__(self, config: LLMConfig):
        """
        Args:
            config: Model configuration containing MLP parameters
        """
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Gate projection: hidden_size → intermediate_size
        # This branch will be activated with Swish
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        
        # Up projection: hidden_size → intermediate_size
        # This branch provides the values to be gated
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        
        # Down projection: intermediate_size → hidden_size
        # Projects back to original dimension
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
        
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Gate branch: apply Swish activation
        # Swish(x) = x * sigmoid(x) = silu(x)
        gate = F.silu(self.gate_proj(x))
        
        # Up branch: no activation
        up = self.up_proj(x)
        
        # Element-wise multiplication (gating)
        hidden = gate * up
        
        # Project back down
        output = self.down_proj(hidden)
        
        return output


class MLP(nn.Module):
    """
    Standard MLP for comparison (not used in modern LLMs but good to understand).
    
    Architecture:
        x ──→ W_up ──→ ReLU ──→ W_down ──→ output
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.relu(self.up_proj(x)))


if __name__ == "__main__":
    from config import LLMConfig
    
    # Create config
    config = LLMConfig(
        hidden_size=768,
        intermediate_size=2048  # Usually ~2.7x hidden_size for SwiGLU
    )
    
    # Create both MLPs
    swiglu = SwiGLU(config)
    standard_mlp = MLP(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    out_swiglu = swiglu(x)
    out_standard = standard_mlp(x)
    
    print(f"Input shape:       {x.shape}")
    print(f"SwiGLU output:     {out_swiglu.shape}")
    print(f"Standard output:   {out_standard.shape}")
    
    # Count parameters
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    standard_params = sum(p.numel() for p in standard_mlp.parameters())
    
    print(f"\nSwiGLU parameters:   {swiglu_params:,}")
    print(f"Standard parameters: {standard_params:,}")
    print(f"SwiGLU has {swiglu_params / standard_params:.1f}x more parameters")
    
    # Visualize the gating
    print("\n--- Gating visualization ---")
    sample = torch.randn(1, 1, config.hidden_size)
    with torch.no_grad():
        gate_values = F.silu(swiglu.gate_proj(sample))
        up_values = swiglu.up_proj(sample)
        gated = gate_values * up_values
    
    print(f"Gate range:   [{gate_values.min():.3f}, {gate_values.max():.3f}]")
    print(f"Up range:     [{up_values.min():.3f}, {up_values.max():.3f}]")
    print(f"Gated range:  [{gated.min():.3f}, {gated.max():.3f}]")