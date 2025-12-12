import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from config import LLMConfig
from positional import RoPE, apply_rope

# GQA is present because K/V have fewer heads(4) than Q(12) and _repeat_kv() duplicates them to match Q head count.

class Attention(nn.Module):
    """
    Multi-Head Attention with support for Grouped Query Attention (GQA).
    
    In standard MHA: num_heads == num_kv_heads (each head has its own K,V)
    In GQA: num_kv_heads < num_heads (multiple Q heads share K,V heads)
    
    This reduces memory usage during inference (smaller KV cache) while
    maintaining most of the model quality.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Args:
            config: Model configuration containing attention parameters
        """
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        # How many Q heads per KV head
        self.num_groups = self.num_heads // self.num_kv_heads
        
        # Projection layers
        # Q has full num_heads, K and V have reduced num_kv_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary position embeddings
        self.rope = RoPE(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta
        )
        
        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
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
            mask: Attention mask (optional) - True/1 for positions to mask
            start_pos: Starting position for RoPE (used in generation)
            kv_cache: Tuple of (cached_k, cached_v) from previous steps
        
        Returns:
            output: Attention output of shape (batch, seq_len, hidden_size)
            new_kv_cache: Updated (k, v) cache for next step
        """
        batch, seq_len, _ = x.shape
        
        # Step 1: Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, num_heads * head_dim)
        k = self.k_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
        
        # Step 2: Reshape to separate heads
        # Q: (batch, seq_len, num_heads, head_dim)
        # K, V: (batch, seq_len, num_kv_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        # Step 3: Apply rotary position embeddings to Q and K
        q, k = apply_rope(q, k, self.rope, start_pos)
        
        # Step 4: Handle KV cache for efficient generation
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            # Concatenate new K, V with cached
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
        
        # Store updated cache
        new_kv_cache = (k, v)
        
        # Step 5: Expand K, V for Grouped Query Attention
        # Repeat K, V heads to match number of Q heads
        if self.num_groups > 1:
            k = self._repeat_kv(k)  # (batch, kv_seq_len, num_heads, head_dim)
            v = self._repeat_kv(v)
        
        # Step 6: Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Step 7: Compute attention scores
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, kv_seq_len)
        # = (batch, num_heads, seq_len, kv_seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Step 8: Apply causal mask (prevent attending to future tokens)
        kv_seq_len = k.shape[2]
        causal_mask = self._make_causal_mask(seq_len, kv_seq_len, start_pos, x.device)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Step 9: Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Step 10: Apply attention to values
        # (batch, num_heads, seq_len, kv_seq_len) @ (batch, num_heads, kv_seq_len, head_dim)
        # = (batch, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)
        
        # Step 11: Reshape back: (batch, seq_len, num_heads * head_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, self.num_heads * self.head_dim)
        
        # Step 12: Final projection
        output = self.o_proj(output)
        
        return output, new_kv_cache
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat K or V heads to match the number of Q heads for GQA.
        
        Args:
            x: Tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        
        Returns:
            Tensor of shape (batch, seq_len, num_heads, head_dim)
        """
        batch, seq_len, num_kv_heads, head_dim = x.shape
        
        # Add a dimension and repeat
        # (batch, seq_len, num_kv_heads, 1, head_dim)
        x = x.unsqueeze(3)
        
        # (batch, seq_len, num_kv_heads, num_groups, head_dim)
        x = x.expand(batch, seq_len, num_kv_heads, self.num_groups, head_dim)
        
        # (batch, seq_len, num_heads, head_dim)
        x = x.reshape(batch, seq_len, self.num_heads, head_dim)
        
        return x
    
    def _make_causal_mask(
        self,
        seq_len: int,
        kv_seq_len: int,
        start_pos: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create causal mask to prevent attending to future tokens.
        
        For training (start_pos=0): Standard lower triangular mask
        For generation (start_pos>0): Only mask future tokens relative to current position
        
        Returns:
            Boolean mask of shape (1, 1, seq_len, kv_seq_len)
            True = mask this position (don't attend)
        """
        # Create position indices
        q_pos = torch.arange(start_pos, start_pos + seq_len, device=device)
        k_pos = torch.arange(kv_seq_len, device=device)
        
        # Mask where key position > query position (future tokens)
        # (seq_len, kv_seq_len)
        mask = q_pos.unsqueeze(1) < k_pos.unsqueeze(0)
        
        # Add batch and head dimensions
        return mask.unsqueeze(0).unsqueeze(0)



if __name__ == "__main__":
    from config import LLMConfig
    
    # Create config
    config = LLMConfig(
        hidden_size=768,
        num_heads=12,
        num_kv_heads=4,  # GQA: 12 Q heads, 4 KV heads
        max_seq_len=2048
    )
    
    # Create attention module
    attn = Attention(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output, kv_cache = attn(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"KV cache K shape: {kv_cache[0].shape}")
    print(f"KV cache V shape: {kv_cache[1].shape}")
    
    # Test generation (with cache)
    print("\n--- Testing generation with KV cache ---")
    new_token = torch.randn(batch_size, 1, config.hidden_size)
    output2, kv_cache2 = attn(new_token, start_pos=seq_len, kv_cache=kv_cache)
    
    print(f"New token shape: {new_token.shape}")
    print(f"Output shape:    {output2.shape}")
    print(f"Updated KV cache K shape: {kv_cache2[0].shape}")
