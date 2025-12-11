import torch
import torch.nn as nn

# Refer Obsidian Notes for Conceptual Understanding

class RoPE(nn.Module):
    """
    Rotary Position Embeddings (RoPE)
    
    Instead of adding position embeddings, RoPE rotates query and key vectors
    based on their position. This encodes relative position information
    directly into the attention mechanism.
    
    The rotation is applied to pairs of dimensions using a rotation matrix,
    where the rotation angle depends on position and dimension.
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        """
        Args:
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length to precompute
            theta: Base for the frequency calculation (10000 is standard)
        """
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute the frequency bands and rotation matrices
        # Register as buffer (saved with model, but not a parameter)
        freqs_cos, freqs_sin = self._precompute_freqs()
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)
    
    def _precompute_freqs(self):
        """
        Precompute cosine and sine frequencies for all positions.
        
        The frequency for dimension i is: theta^(-2i/head_dim)
        This gives lower frequencies (slower rotation) for later dimensions.
        """
        # Step 1: Compute frequency for each pair of dimensions
        # Shape: (head_dim // 2,)
        dim_indices = torch.arange(0, self.head_dim, 2).float()
        freqs = 1.0 / (self.theta ** (dim_indices / self.head_dim))
        
        # Step 2: Compute position indices
        # Shape: (max_seq_len,)
        positions = torch.arange(self.max_seq_len).float()
        
        # Step 3: Outer product to get angles for each (position, dimension) pair
        # Shape: (max_seq_len, head_dim // 2)
        angles = torch.outer(positions, freqs)
        
        # Step 4: Compute cos and sin
        # Shape: (max_seq_len, head_dim // 2)
        freqs_cos = torch.cos(angles)
        freqs_sin = torch.sin(angles)
        
        return freqs_cos, freqs_sin
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
            start_pos: Starting position (for caching during generation)
        
        Returns:
            Rotated tensor of same shape
        """
        batch, seq_len, num_heads, head_dim = x.shape
        
        # Get the relevant frequencies for this sequence
        # Shape: (seq_len, head_dim // 2)
        cos = self.freqs_cos[start_pos : start_pos + seq_len]
        sin = self.freqs_sin[start_pos : start_pos + seq_len]
        
        # Reshape for broadcasting: (seq_len, 1, head_dim // 2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        # Split x into pairs of dimensions for rotation
        # Shape: (batch, seq_len, num_heads, head_dim // 2, 2)
        x_pairs = x.view(batch, seq_len, num_heads, head_dim // 2, 2)
        
        # Separate the pairs
        x_even = x_pairs[..., 0]  # (batch, seq_len, num_heads, head_dim // 2)
        x_odd = x_pairs[..., 1]   # (batch, seq_len, num_heads, head_dim // 2)
        
        # Apply rotation using rotation matrix:
        # [cos, -sin] [x_even]   [x_even * cos - x_odd * sin]
        # [sin,  cos] [x_odd ] = [x_even * sin + x_odd * cos]
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        
        # Interleave back together
        # Stack and reshape back to original shape
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        x_rot = x_rot.view(batch, seq_len, num_heads, head_dim)
        
        return x_rot


def apply_rope(q: torch.Tensor, k: torch.Tensor, rope: RoPE, start_pos: int = 0):
    """
    Convenience function to apply RoPE to both query and key tensors.
    
    Args:
        q: Query tensor (batch, seq_len, num_heads, head_dim)
        k: Key tensor (batch, seq_len, num_kv_heads, head_dim)
        rope: RoPE module
        start_pos: Starting position for caching
    
    Returns:
        Rotated (q, k) tensors
    """
    q_rot = rope(q, start_pos)
    k_rot = rope(k, start_pos)
    return q_rot, k_rot


if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    num_heads = 12
    head_dim = 64
    
    # Create RoPE module
    rope = RoPE(head_dim=head_dim, max_seq_len=2048)
    
    # Create sample query and key tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    # Apply RoPE
    q_rot, k_rot = apply_rope(q, k, rope)
    
    print(f"Query shape:         {q.shape}")
    print(f"Rotated query shape: {q_rot.shape}")
    print(f"Freqs cos shape:     {rope.freqs_cos.shape}")
    
    # Verify: rotation preserves vector magnitude
    q_norm = q.norm(dim=-1).mean()
    q_rot_norm = q_rot.norm(dim=-1).mean()
    print(f"Original Q norm: {q_norm:.4f}")
    print(f"Rotated Q norm:  {q_rot_norm:.4f}")
