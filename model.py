import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import LLMConfig
from normalization import RMSNorm
from decoder import DecoderLayer


class LLM(nn.Module):
    """
    Full Large Language Model.
    
    This is a decoder-only transformer that:
    1. Embeds input tokens into continuous vectors
    2. Processes through a stack of decoder layers
    3. Projects back to vocabulary size for next-token prediction
    
    Architecture:
        tokens → Embedding → [DecoderLayer × num_layers] → RMSNorm → Linear → logits
    """
    
    def __init__(self, config: LLMConfig):
        """
        Args:
            config: Model configuration with all hyperparameters
        """
        super().__init__()
        self.config = config
        
        # Token embedding: vocab_size → hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Final normalization (before output projection)
        self.final_norm = RMSNorm(config.hidden_size)
        
        # Output projection: hidden_size → vocab_size
        # This produces logits for next-token prediction
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Optionally tie input/output embeddings (saves parameters)
        # Many modern LLMs do this
        if getattr(config, 'tie_embeddings', True):
            self.output_proj.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
        
        # Report model size
        self.num_params = self._count_parameters()
    
    def _init_weights(self):
        """
        Initialize weights using standard practices.
        
        - Embeddings: Normal(0, 0.02)
        - Linear layers: Normal(0, 0.02)
        - Some layers scaled by 1/sqrt(2*num_layers) for residual paths
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[list] = None,
        return_kv_cache: bool = False
    ) -> dict:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            mask: Optional attention mask
            start_pos: Starting position for generation (used with KV cache)
            kv_cache: List of (k, v) tuples for each layer (for generation)
            return_kv_cache: Whether to return updated KV cache
        
        Returns:
            Dictionary containing:
                - logits: Output logits of shape (batch, seq_len, vocab_size)
                - kv_cache: Updated KV cache (if return_kv_cache=True)
        """
        batch, seq_len = input_ids.shape
        
        # Step 1: Token embedding
        # (batch, seq_len) → (batch, seq_len, hidden_size)
        x = self.token_embedding(input_ids)
        
        # Initialize KV cache if not provided
        if kv_cache is None:
            kv_cache = [None] * self.config.num_layers
        
        # Step 2: Process through decoder layers
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            x, layer_kv = layer(
                x,
                mask=mask,
                start_pos=start_pos,
                kv_cache=kv_cache[i]
            )
            new_kv_cache.append(layer_kv)
        
        # Step 3: Final normalization
        x = self.final_norm(x)
        
        # Step 4: Project to vocabulary
        # (batch, seq_len, hidden_size) → (batch, seq_len, vocab_size)
        logits = self.output_proj(x)
        
        # Build output dictionary
        output = {"logits": logits}
        if return_kv_cache:
            output["kv_cache"] = new_kv_cache
        
        return output
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for language modeling.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            target_ids: Target token IDs (batch, seq_len)
            mask: Optional mask for ignoring certain positions
        
        Returns:
            Scalar loss tensor
        """
        # Forward pass
        output = self.forward(input_ids)
        logits = output["logits"]
        
        # Reshape for cross-entropy
        # logits: (batch * seq_len, vocab_size)
        # targets: (batch * seq_len,)
        batch, seq_len, vocab_size = logits.shape
        logits = logits.view(batch * seq_len, vocab_size)
        targets = target_ids.view(batch * seq_len)
        
        # Compute loss (ignore padding tokens, usually -100)
        loss = F.cross_entropy(logits, targets, ignore_index=-100)
        
        return loss
    
    @torch.no_grad()
    def generate_next_token(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        start_pos: int = 0,
        kv_cache: Optional[list] = None
    ) -> tuple[torch.Tensor, list]:
        """
        Generate the next token given input.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens
            top_p: Keep tokens with cumulative probability <= p
            start_pos: Position for KV cache
            kv_cache: Cached key/values from previous steps
        
        Returns:
            next_token: The sampled token (batch, 1)
            new_kv_cache: Updated KV cache
        """
        # Forward pass
        output = self.forward(
            input_ids,
            start_pos=start_pos,
            kv_cache=kv_cache,
            return_kv_cache=True
        )
        
        # Get logits for the last position only
        logits = output["logits"][:, -1, :]  # (batch, vocab_size)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift to keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter back to original order
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        
        return next_token, output["kv_cache"]
    
    def __repr__(self):
        """Pretty print model info."""
        return (
            f"LLM(\n"
            f"  vocab_size={self.config.vocab_size:,}\n"
            f"  hidden_size={self.config.hidden_size}\n"
            f"  num_layers={self.config.num_layers}\n"
            f"  num_heads={self.config.num_heads}\n"
            f"  num_kv_heads={self.config.num_kv_heads}\n"
            f"  parameters={self.num_params:,}\n"
            f")"
        )
    


if __name__ == "__main__":
    from config import LLMConfig
    
    # Create a small config for testing
    config = LLMConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        num_kv_heads=4,
        intermediate_size=2048,
        max_seq_len=2048
    )
    
    # Create model
    model = LLM(config)
    print(model)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = model(input_ids)
    logits = output["logits"]
    
    print(f"\nInput shape:  {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test loss computation
    target_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    loss = model.compute_loss(input_ids, target_ids)
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    print("\n--- Generation test ---")
    prompt = torch.randint(0, config.vocab_size, (1, 5))  # Single sequence, 5 tokens
    print(f"Prompt shape: {prompt.shape}")
    
    # Generate with KV cache
    kv_cache = None
    generated = prompt.clone()
    
    for i in range(5):  # Generate 5 tokens
        if kv_cache is None:
            # First pass: process entire prompt
            next_token, kv_cache = model.generate_next_token(
                generated,
                temperature=0.8,
                top_k=50
            )
        else:
            # Subsequent passes: only process new token
            next_token, kv_cache = model.generate_next_token(
                generated[:, -1:],  # Only last token
                start_pos=generated.shape[1] - 1,
                kv_cache=kv_cache,
                temperature=0.8,
                top_k=50
            )
        
        generated = torch.cat([generated, next_token], dim=1)
        print(f"Step {i+1}: generated token {next_token.item()}, total length: {generated.shape[1]}")
    
    print(f"\nFinal generated shape: {generated.shape}")
    
    # Model size comparison
    print("\n--- Model size estimates ---")
    configs = [
        ("Tiny (this test)", 768, 12),
        ("Small (~125M)", 768, 12),
        ("Medium (~350M)", 1024, 24),
        ("Large (~760M)", 1536, 24),
        ("XL (~1.5B)", 2048, 24),
    ]
    
    for name, hidden, layers in configs:
        params = hidden * 32000  # embedding
        params += layers * (4 * hidden * hidden + 3 * hidden * 2048)  # rough estimate
        print(f"{name}: ~{params/1e6:.0f}M parameters")
