import torch
import torch.nn as nn
from typing import Optional

from config import LLMConfig
from normalization import RMSNorm
from attention import Attention
from mlp import SwiGLU


class DecoderLayer(nn.Module):
    """
    A single transformer decoder layer.
    
    This is the fundamental building block of the LLM. Each layer:
    1. Applies self-attention (tokens look at each other)
    2. Applies feed-forward network (process each token)
    
    Uses Pre-Norm architecture (norm before each sub-layer) which is
    more stable for training deep networks.
    
    Architecture:
        x → RMSNorm → Attention → + (residual) → RMSNorm → MLP → + (residual) → output
    """
    
    def __init__(self, config: LLMConfig, layer_idx: int):
        """
        Args:
            config: Model configuration
            layer_idx: Index of this layer (useful for debugging/analysis)
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Pre-attention normalization
        self.attn_norm = RMSNorm(config.hidden_size)
        
        # Self-attention
        self.attention = Attention(config)
        
        # Pre-MLP normalization
        self.mlp_norm = RMSNorm(config.hidden_size)
        
        # Feed-forward network
        self.mlp = SwiGLU(config)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[tuple] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            mask: Optional attention mask
            start_pos: Starting position for RoPE (used in generation)
            kv_cache: Optional KV cache from previous forward passes
        
        Returns:
            output: Output tensor of shape (batch, seq_len, hidden_size)
            new_kv_cache: Updated KV cache
        """
        # ============ Attention Block ============
        # Pre-norm
        normed = self.attn_norm(x)
        
        # Self-attention
        attn_output, new_kv_cache = self.attention(
            normed,
            mask=mask,
            start_pos=start_pos,
            kv_cache=kv_cache
        )
        
        # Residual connection
        x = x + attn_output
        
        # ============ MLP Block ============
        # Pre-norm
        normed = self.mlp_norm(x)
        
        # Feed-forward
        mlp_output = self.mlp(normed)
        
        # Residual connection
        x = x + mlp_output
        
        return x, new_kv_cache


class DecoderLayerWithDropout(DecoderLayer):
    """
    Decoder layer with dropout for training.
    
    Dropout is applied after attention and after MLP, before the residual add.
    This helps prevent overfitting during training.
    """
    
    def __init__(self, config: LLMConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[tuple] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Same as parent but with dropout."""
        
        # ============ Attention Block ============
        normed = self.attn_norm(x)
        attn_output, new_kv_cache = self.attention(
            normed,
            mask=mask,
            start_pos=start_pos,
            kv_cache=kv_cache
        )
        # Dropout before residual
        x = x + self.dropout(attn_output)
        
        # ============ MLP Block ============
        normed = self.mlp_norm(x)
        mlp_output = self.mlp(normed)
        # Dropout before residual
        x = x + self.dropout(mlp_output)
        
        return x, new_kv_cache


if __name__ == "__main__":
    from config import LLMConfig
    
    # Create config
    config = LLMConfig(
        hidden_size=768,
        num_heads=12,
        num_kv_heads=4,
        intermediate_size=2048,
        max_seq_len=2048,
        dropout=0.1
    )
    
    # Create a decoder layer
    layer = DecoderLayer(config, layer_idx=0)
    layer_with_dropout = DecoderLayerWithDropout(config, layer_idx=0)
    
    # Test input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output, kv_cache = layer(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"KV cache K:   {kv_cache[0].shape}")
    print(f"KV cache V:   {kv_cache[1].shape}")
    
    # Verify residual connection is working
    # Output should be different from input but in same ballpark
    print(f"\nInput mean:  {x.mean():.4f}, std: {x.std():.4f}")
    print(f"Output mean: {output.mean():.4f}, std: {output.std():.4f}")
    
    # Test generation mode (single token with cache)
    print("\n--- Generation mode ---")
    new_token = torch.randn(batch_size, 1, config.hidden_size)
    output2, kv_cache2 = layer(new_token, start_pos=seq_len, kv_cache=kv_cache)
    
    print(f"New token shape: {new_token.shape}")
    print(f"Output shape:    {output2.shape}")
    print(f"KV cache K now:  {kv_cache2[0].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in layer.parameters())
    print(f"\nParameters per layer: {total_params:,}")
    print(f"For 12 layers: {total_params * 12:,}")
