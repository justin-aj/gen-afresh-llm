import torch
import torch.nn.functional as F
from typing import Optional, List, Union
from dataclasses import dataclass
import time

from config import LLMConfig
from model import LLM
from tokenizer import BaseTokenizer, get_tokenizer


# ============================================================
# Generation Configuration
# ============================================================

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    # Sampling parameters
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    
    # Generation limits
    max_new_tokens: int = 100
    
    # Stop conditions
    stop_tokens: Optional[List[int]] = None  # Stop when these tokens appear
    
    # Repetition penalty
    repetition_penalty: float = 1.0  # 1.0 = no penalty
    
    # Output options
    stream: bool = False  # Print tokens as they're generated


# ============================================================
# Text Generator
# ============================================================

class TextGenerator:
    """
    Text generation using a trained LLM.
    
    Supports various sampling strategies and efficient generation
    using KV caching.
    """
    
    def __init__(
        self,
        model: LLM,
        tokenizer: BaseTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model: Trained LLM
            tokenizer: Tokenizer matching the model
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text to continue from
            config: Generation configuration
        
        Returns:
            Generated text (including prompt)
        """
        if config is None:
            config = GenerationConfig()
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Track generated tokens
        generated_ids = input_ids.clone()
        
        # Initialize KV cache
        kv_cache = None
        
        # Generation loop
        start_time = time.time()
        
        for i in range(config.max_new_tokens):
            # Get next token
            next_token, kv_cache = self._generate_next_token(
                generated_ids if kv_cache is None else generated_ids[:, -1:],
                config,
                start_pos=0 if kv_cache is None else generated_ids.shape[1] - 1,
                kv_cache=kv_cache,
                all_generated_ids=generated_ids
            )
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stream output
            if config.stream:
                token_text = self.tokenizer.decode([next_token.item()])
                print(token_text, end="", flush=True)
            
            # Check stop conditions
            if self._should_stop(next_token, config):
                break
        
        if config.stream:
            print()  # Newline after streaming
        
        # Decode full sequence
        output_ids = generated_ids[0].tolist()
        output_text = self.tokenizer.decode(output_ids)
        
        # Stats
        elapsed = time.time() - start_time
        tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
        
        return output_text
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts in parallel.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration
        
        Returns:
            List of generated texts
        """
        if config is None:
            config = GenerationConfig()
        
        # Encode all prompts
        batch_ids = []
        max_len = 0
        
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            batch_ids.append(ids)
            max_len = max(max_len, len(ids))
        
        # Pad to same length
        pad_id = self.tokenizer.pad_token_id
        padded_ids = []
        attention_mask = []
        
        for ids in batch_ids:
            padding = [pad_id] * (max_len - len(ids))
            padded_ids.append(padding + ids)  # Left padding
            attention_mask.append([0] * len(padding) + [1] * len(ids))
        
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        
        # Generation loop (without KV cache for simplicity in batch mode)
        generated_ids = input_ids.clone()
        
        for i in range(config.max_new_tokens):
            # Forward pass
            output = self.model(generated_ids, return_kv_cache=False)
            logits = output["logits"][:, -1, :]
            
            # Sample next tokens
            next_tokens = self._sample_tokens(logits, config, generated_ids)
            
            # Append
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)
        
        # Decode all sequences
        outputs = []
        for i, ids in enumerate(generated_ids.tolist()):
            # Remove padding
            ids = [t for t in ids if t != pad_id]
            outputs.append(self.tokenizer.decode(ids))
        
        return outputs
    
    def _generate_next_token(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        start_pos: int = 0,
        kv_cache: Optional[list] = None,
        all_generated_ids: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, list]:
        """Generate the next token."""
        
        # Forward pass
        output = self.model(
            input_ids,
            start_pos=start_pos,
            kv_cache=kv_cache,
            return_kv_cache=True
        )
        
        logits = output["logits"][:, -1, :]  # (batch, vocab_size)
        
        # Apply repetition penalty
        if config.repetition_penalty != 1.0 and all_generated_ids is not None:
            logits = self._apply_repetition_penalty(
                logits, all_generated_ids, config.repetition_penalty
            )
        
        # Sample
        next_token = self._sample_tokens(logits, config)
        
        return next_token.unsqueeze(1), output["kv_cache"]
    
    def _sample_tokens(
        self,
        logits: torch.Tensor,
        config: GenerationConfig,
        generated_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample tokens from logits using configured strategy."""
        
        # Apply temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature
        
        # Apply top-k filtering
        if config.top_k is not None:
            logits = self._top_k_filtering(logits, config.top_k)
        
        # Apply top-p (nucleus) filtering
        if config.top_p is not None:
            logits = self._top_p_filtering(logits, config.top_p)
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return next_tokens
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Keep only top k tokens."""
        top_k = min(top_k, logits.size(-1))
        
        # Find the top-k threshold
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        
        # Mask tokens below threshold
        logits = torch.where(
            logits < threshold,
            torch.full_like(logits, float('-inf')),
            logits
        )
        
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Keep smallest set of tokens with cumulative prob >= top_p."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Find cutoff
        sorted_mask = cumulative_probs > top_p
        # Keep at least one token
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        
        # Scatter mask back to original order
        mask = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
        
        logits = torch.where(
            mask,
            torch.full_like(logits, float('-inf')),
            logits
        )
        
        return logits
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Reduce probability of already-generated tokens."""
        # Get unique tokens in generated sequence
        for i in range(logits.size(0)):
            unique_tokens = generated_ids[i].unique()
            
            # Penalize these tokens
            for token in unique_tokens:
                if logits[i, token] > 0:
                    logits[i, token] /= penalty
                else:
                    logits[i, token] *= penalty
        
        return logits
    
    def _should_stop(self, token: torch.Tensor, config: GenerationConfig) -> bool:
        """Check if generation should stop."""
        token_id = token.item()
        
        # Stop on EOS
        if token_id == self.tokenizer.eos_token_id:
            return True
        
        # Stop on custom stop tokens
        if config.stop_tokens and token_id in config.stop_tokens:
            return True
        
        return False


# ============================================================
# Interactive Chat
# ============================================================

def interactive_chat(generator: TextGenerator, config: GenerationConfig):
    """Simple interactive chat loop."""
    print("=" * 60)
    print("Interactive Chat (type 'quit' to exit)")
    print("=" * 60)
    print()
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("Model: ", end="")
            
            # Generate with streaming
            stream_config = GenerationConfig(
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                max_new_tokens=config.max_new_tokens,
                stream=True
            )
            
            output = generator.generate(prompt, stream_config)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


# ============================================================
# Utility Functions
# ============================================================

def load_model_for_generation(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple[LLM, LLMConfig]:
    """
    Load a trained model from checkpoint.
    
    Returns:
        model, config
    """
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    model_config = LLMConfig(**checkpoint["model_config"])
    
    # Create and load model
    model = LLM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model with {model.num_params:,} parameters")
    
    return model, model_config


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Generate text with trained LLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Input prompt (if not provided, enters interactive mode)")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Training a quick demo model...")
        
        # Create a tiny model for demo
        from tokenizer import SimpleTokenizer
        
        model_config = LLMConfig(
            vocab_size=256,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=256,
            max_seq_len=128,
        )
        
        model = LLM(model_config)
        tokenizer = SimpleTokenizer()
        
        print("Using untrained model (output will be random)")
    else:
        # Load trained model
        model, model_config = load_model_for_generation(args.checkpoint, device)
        
        # Use simple tokenizer (should match training)
        from tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer()
    
    # Create generator
    generator = TextGenerator(model, tokenizer, device)
    
    # Generation config
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens,
    )
    
    if args.prompt:
        # Single generation
        print(f"\nPrompt: {args.prompt}")
        print("-" * 40)
        
        gen_config.stream = True
        print("Output: ", end="")
        output = generator.generate(args.prompt, gen_config)
        print()
    else:
        # Interactive mode
        interactive_chat(generator, gen_config)