import os
import json
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

from config import LLMConfig
from model import LLM


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic operations (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


# ============================================================
# Device Management
# ============================================================

def get_device(device: str = "auto") -> torch.device:
    """
    Get the best available device.
    
    Args:
        device: "auto", "cuda", "cpu", or specific like "cuda:0"
    
    Returns:
        torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
        else:
            device = "cpu"
    
    return torch.device(device)


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    
    if info["cuda_available"]:
        info["cuda_devices"] = []
        for i in range(info["cuda_device_count"]):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append({
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    
    return info


# ============================================================
# Model Analysis
# ============================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dictionary with total, trainable, and non-trainable counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }


def estimate_memory(config: LLMConfig, batch_size: int = 1, seq_len: int = 512) -> Dict[str, float]:
    """
    Estimate memory usage for a model configuration.
    
    Returns:
        Dictionary with memory estimates in GB
    """
    # Model parameters (4 bytes per fp32, 2 bytes per fp16)
    num_params = (
        config.vocab_size * config.hidden_size +  # Embedding
        config.num_layers * (
            4 * config.hidden_size * config.hidden_size +  # Attention (Q, K, V, O)
            3 * config.hidden_size * config.intermediate_size  # MLP (gate, up, down)
        ) +
        config.hidden_size * config.vocab_size  # Output projection
    )
    
    params_fp32 = num_params * 4 / (1024**3)
    params_fp16 = num_params * 2 / (1024**3)
    
    # Activations (rough estimate)
    # Each layer stores: attention output, MLP output, residuals
    activations_per_layer = batch_size * seq_len * config.hidden_size * 4 * 3
    total_activations = config.num_layers * activations_per_layer / (1024**3)
    
    # KV Cache (for generation)
    kv_cache_per_layer = 2 * batch_size * seq_len * config.num_kv_heads * (config.hidden_size // config.num_heads) * 2
    total_kv_cache = config.num_layers * kv_cache_per_layer / (1024**3)
    
    # Optimizer states (AdamW: 2x for momentum and variance)
    optimizer_states = num_params * 4 * 2 / (1024**3)
    
    return {
        "parameters_fp32_gb": params_fp32,
        "parameters_fp16_gb": params_fp16,
        "activations_gb": total_activations,
        "kv_cache_gb": total_kv_cache,
        "optimizer_states_gb": optimizer_states,
        "training_total_fp32_gb": params_fp32 + total_activations + optimizer_states,
        "training_total_fp16_gb": params_fp16 + total_activations / 2 + optimizer_states,
        "inference_fp16_gb": params_fp16 + total_kv_cache,
    }


def print_model_summary(model: LLM, config: LLMConfig):
    """Print a nice summary of the model."""
    params = count_parameters(model)
    memory = estimate_memory(config)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"\nArchitecture:")
    print(f"  Vocabulary size:    {config.vocab_size:,}")
    print(f"  Hidden size:        {config.hidden_size}")
    print(f"  Num layers:         {config.num_layers}")
    print(f"  Num attention heads:{config.num_heads}")
    print(f"  Num KV heads:       {config.num_kv_heads}")
    print(f"  MLP intermediate:   {config.intermediate_size}")
    print(f"  Max sequence length:{config.max_seq_len}")
    
    print(f"\nParameters:")
    print(f"  Total:              {params['total']:,} ({params['total_millions']:.2f}M)")
    print(f"  Trainable:          {params['trainable']:,}")
    
    print(f"\nMemory Estimates:")
    print(f"  Model (fp16):       {memory['parameters_fp16_gb']:.2f} GB")
    print(f"  Training (fp16):    {memory['training_total_fp16_gb']:.2f} GB")
    print(f"  Inference (fp16):   {memory['inference_fp16_gb']:.2f} GB")
    print("=" * 60)


def get_layer_info(model: LLM) -> List[Dict[str, Any]]:
    """Get detailed information about each layer."""
    layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            params = sum(p.numel() for p in module.parameters())
            layers.append({
                "name": name,
                "type": type(module).__name__,
                "parameters": params,
                "shape": list(module.weight.shape) if hasattr(module, "weight") else None,
            })
    
    return layers


# ============================================================
# Weight Loading (for pretrained models)
# ============================================================

def load_pretrained_weights(
    model: LLM,
    weights_path: str,
    key_mapping: Optional[Dict[str, str]] = None,
    strict: bool = False
) -> List[str]:
    """
    Load pretrained weights into model.
    
    Args:
        model: Model to load weights into
        weights_path: Path to weights file (.pt, .bin, .safetensors)
        key_mapping: Optional mapping from pretrained keys to model keys
        strict: If True, raise error on missing/unexpected keys
    
    Returns:
        List of keys that weren't loaded
    """
    print(f"Loading weights from {weights_path}...")
    
    # Load based on file type
    if weights_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        except ImportError:
            raise ImportError("Please install safetensors: pip install safetensors")
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # Handle nested state dicts
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    
    # Apply key mapping
    if key_mapping:
        new_state_dict = {}
        for old_key, value in state_dict.items():
            new_key = key_mapping.get(old_key, old_key)
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    
    if missing:
        print(f"Missing keys: {len(missing)}")
        for key in missing[:5]:
            print(f"  - {key}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for key in unexpected[:5]:
            print(f"  - {key}")
        if len(unexpected) > 5:
            print(f"  ... and {len(unexpected) - 5} more")
    
    return missing + unexpected


def get_llama_key_mapping() -> Dict[str, str]:
    """
    Get key mapping from LLaMA weight names to our model.
    
    Use this when loading official LLaMA weights.
    """
    # This is a template - actual mapping depends on LLaMA version
    mapping = {
        "tok_embeddings.weight": "token_embedding.weight",
        "norm.weight": "final_norm.weight",
        "output.weight": "output_proj.weight",
    }
    
    # Layer mappings
    for i in range(100):  # Support up to 100 layers
        mapping.update({
            f"layers.{i}.attention_norm.weight": f"layers.{i}.attn_norm.weight",
            f"layers.{i}.ffn_norm.weight": f"layers.{i}.mlp_norm.weight",
            f"layers.{i}.attention.wq.weight": f"layers.{i}.attention.q_proj.weight",
            f"layers.{i}.attention.wk.weight": f"layers.{i}.attention.k_proj.weight",
            f"layers.{i}.attention.wv.weight": f"layers.{i}.attention.v_proj.weight",
            f"layers.{i}.attention.wo.weight": f"layers.{i}.attention.o_proj.weight",
            f"layers.{i}.feed_forward.w1.weight": f"layers.{i}.mlp.gate_proj.weight",
            f"layers.{i}.feed_forward.w2.weight": f"layers.{i}.mlp.down_proj.weight",
            f"layers.{i}.feed_forward.w3.weight": f"layers.{i}.mlp.up_proj.weight",
        })
    
    return mapping


# ============================================================
# Training Logger
# ============================================================

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    loss: float
    learning_rate: float
    tokens_per_second: float
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class TrainingLogger:
    """
    Logger for tracking and saving training metrics.
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Name for this experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.metrics: List[TrainingMetrics] = []
        
        # Setup file logging
        log_file = self.log_dir / f"{experiment_name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)
    
    def log_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        tokens_per_second: float,
        **extra
    ):
        """Log a training step."""
        metrics = TrainingMetrics(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            tokens_per_second=tokens_per_second
        )
        self.metrics.append(metrics)
        
        msg = f"Step {step:6d} | Loss {loss:.4f} | LR {learning_rate:.2e} | Tok/s {tokens_per_second:.0f}"
        if extra:
            msg += " | " + " | ".join(f"{k}={v}" for k, v in extra.items())
        
        self.logger.info(msg)
    
    def log_eval(self, step: int, val_loss: float, **extra):
        """Log evaluation results."""
        msg = f"[EVAL] Step {step:6d} | Val Loss {val_loss:.4f}"
        if extra:
            msg += " | " + " | ".join(f"{k}={v}" for k, v in extra.items())
        
        self.logger.info(msg)
    
    def save_metrics(self):
        """Save all metrics to JSON file."""
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        
        with open(metrics_file, "w") as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)
        
        self.logger.info(f"Saved metrics to {metrics_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics:
            return {}
        
        losses = [m.loss for m in self.metrics]
        
        return {
            "total_steps": len(self.metrics),
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "avg_loss": sum(losses) / len(losses),
            "avg_tokens_per_second": sum(m.tokens_per_second for m in self.metrics) / len(self.metrics),
        }


# ============================================================
# Configuration Management
# ============================================================

def save_config(config: LLMConfig, path: str):
    """Save configuration to JSON file."""
    with open(path, "w") as f:
        json.dump(config.__dict__, f, indent=2)
    print(f"Saved config to {path}")


def load_config(path: str) -> LLMConfig:
    """Load configuration from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return LLMConfig(**data)


# ============================================================
# Misc Utilities
# ============================================================

def format_number(n: int) -> str:
    """Format large numbers nicely (e.g., 1.5B, 350M, 7K)."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(n)


def get_num_flops(config: LLMConfig, batch_size: int, seq_len: int) -> int:
    """
    Estimate FLOPs for a forward pass.
    
    Rough estimate based on matrix multiplications.
    """
    # Attention FLOPs per layer
    attn_flops = (
        4 * batch_size * seq_len * config.hidden_size * config.hidden_size +  # Q, K, V, O projections
        2 * batch_size * config.num_heads * seq_len * seq_len * (config.hidden_size // config.num_heads)  # Attention
    )
    
    # MLP FLOPs per layer
    mlp_flops = 3 * 2 * batch_size * seq_len * config.hidden_size * config.intermediate_size
    
    # Total
    total_flops = config.num_layers * (attn_flops + mlp_flops)
    
    # Embedding and output
    total_flops += 2 * batch_size * seq_len * config.hidden_size * config.vocab_size
    
    return total_flops


def benchmark_model(
    model: LLM,
    config: LLMConfig,
    device: str = "cuda",
    batch_size: int = 1,
    seq_len: int = 512,
    num_iterations: int = 10,
    warmup_iterations: int = 3
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    print(f"Warming up ({warmup_iterations} iterations)...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_ids)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            
            if device == "cuda":
                start.record()
            else:
                import time
                start_time = time.perf_counter()
            
            _ = model(input_ids)
            
            if device == "cuda":
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                times.append((time.perf_counter() - start_time) * 1000)
    
    avg_time = sum(times) / len(times)
    tokens_per_second = batch_size * seq_len / (avg_time / 1000)
    
    flops = get_num_flops(config, batch_size, seq_len)
    tflops = flops / (avg_time / 1000) / 1e12
    
    return {
        "avg_latency_ms": avg_time,
        "min_latency_ms": min(times),
        "max_latency_ms": max(times),
        "tokens_per_second": tokens_per_second,
        "tflops": tflops,
    }


if __name__ == "__main__":
    from config import LLMConfig
    from model import LLM
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device info
    print("=== Device Info ===")
    device_info = get_device_info()
    print(json.dumps(device_info, indent=2))
    
    # Create a model
    print("\n=== Creating Model ===")
    config = LLMConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        num_kv_heads=4,
        intermediate_size=2048,
        max_seq_len=2048,
    )
    
    model = LLM(config)
    
    # Print summary
    print_model_summary(model, config)
    
    # Memory estimates
    print("\n=== Memory Estimates ===")
    memory = estimate_memory(config, batch_size=8, seq_len=512)
    for key, value in memory.items():
        print(f"  {key}: {value:.2f} GB")
    
    # Layer info
    print("\n=== Layer Info (first 5) ===")
    layers = get_layer_info(model)
    for layer in layers[:5]:
        print(f"  {layer['name']}: {layer['type']} - {format_number(layer['parameters'])} params")
    
    # Save/load config
    print("\n=== Config Management ===")
    save_config(config, "test_config.json")
    loaded_config = load_config("test_config.json")
    print(f"Loaded config: hidden_size={loaded_config.hidden_size}")
    
    # Cleanup
    os.remove("test_config.json")
    
    # Training logger demo
    print("\n=== Training Logger Demo ===")
    logger = TrainingLogger("logs", "demo_experiment")
    
    for step in range(1, 6):
        logger.log_step(
            step=step,
            loss=5.0 - step * 0.5,
            learning_rate=1e-4,
            tokens_per_second=10000
        )
    
    summary = logger.get_summary()
    print(f"Summary: {summary}")
    
    # Benchmark (if GPU available)
    device = get_device()
    print(f"\n=== Benchmark on {device} ===")
    
    if str(device) == "cuda":
        results = benchmark_model(model, config, device="cuda", num_iterations=5)
        for key, value in results.items():
            print(f"  {key}: {value:.2f}")
    else:
        print("  Skipping benchmark (no GPU)")
    
    print("\n✅ All utils working!")

