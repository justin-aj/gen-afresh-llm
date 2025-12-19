import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Callable
from pathlib import Path
from dataclasses import dataclass

from config import LLMConfig
from model import LLM
from tokenizer import get_tokenizer, BaseTokenizer
from datasets import load_dataset
from dataset import HFTextDataset
from torch.utils.data import Dataset as TorchDataset



# ============================================================
# Training Configuration
# ============================================================

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Data
    data_path: str = "data/train.txt"
    val_path: Optional[str] = None
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch = batch_size * this
    max_steps: int = 10000
    
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # LR Schedule
    warmup_steps: int = 100
    min_lr: float = 1e-5
    
    # Logging & Saving
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.
    
    Loads a text file, tokenizes it, and serves chunks of tokens.
    Each chunk is (input, target) where target is input shifted by 1.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: BaseTokenizer,
        max_seq_len: int,
        stride: Optional[int] = None
    ):
        """
        Args:
            file_path: Path to text file
            tokenizer: Tokenizer to use
            max_seq_len: Maximum sequence length
            stride: Step size between chunks (default: max_seq_len // 2)
        """
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len // 2
        
        # Load and tokenize
        print(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        print(f"Tokenizing {len(text):,} characters...")
        self.tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        print(f"Got {len(self.tokens):,} tokens")
        
        # Calculate number of samples
        self.num_samples = max(1, (len(self.tokens) - max_seq_len) // self.stride)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            input_ids: Tokens for input (seq_len,)
            target_ids: Tokens for target (seq_len,) — shifted by 1
        """
        start = idx * self.stride
        end = start + self.max_seq_len + 1  # +1 for target shift
        
        chunk = self.tokens[start:end]
        
        # Pad if needed (for last chunk)
        if len(chunk) < self.max_seq_len + 1:
            chunk = chunk + [0] * (self.max_seq_len + 1 - len(chunk))
        
        chunk = torch.tensor(chunk, dtype=torch.long)
        
        input_ids = chunk[:-1]   # All but last
        target_ids = chunk[1:]   # All but first (shifted by 1)
        
        return input_ids, target_ids


# ============================================================
# Learning Rate Schedule
# ============================================================

def get_lr(step: int, config: TrainConfig) -> float:
    """
    Learning rate schedule with linear warmup and cosine decay.
    
    1. Linear warmup from 0 to max_lr
    2. Cosine decay from max_lr to min_lr
    """
    # Warmup phase
    if step < config.warmup_steps:
        return config.learning_rate * (step / config.warmup_steps)
    
    # Cosine decay phase
    decay_steps = config.max_steps - config.warmup_steps
    step_in_decay = step - config.warmup_steps
    
    # Cosine decay from 1 to 0
    decay_ratio = (1 + math.cos(math.pi * step_in_decay / decay_steps)) / 2
    
    # Scale between min_lr and learning_rate
    return config.min_lr + (config.learning_rate - config.min_lr) * decay_ratio


# ============================================================
# Training Functions
# ============================================================

def train_step(
    model: LLM,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """
    Single training step.
    
    Returns:
        loss value (float)
    """
    input_ids, target_ids = batch
    input_ids = input_ids.to(config.device)
    target_ids = target_ids.to(config.device)
    
    # Forward pass (with optional mixed precision)
    if scaler is not None:
        with torch.cuda.amp.autocast():
            loss = model.compute_loss(input_ids, target_ids)
    else:
        loss = model.compute_loss(input_ids, target_ids)
    
    # Scale loss for gradient accumulation
    loss = loss / config.gradient_accumulation_steps
    
    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    return loss.item() * config.gradient_accumulation_steps


@torch.no_grad()
def evaluate(
    model: LLM,
    dataloader: DataLoader,
    config: TrainConfig,
    max_batches: int = 50
) -> float:
    """
    Evaluate model on validation set.
    
    Returns:
        average loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        if num_batches >= max_batches:
            break
        
        input_ids, target_ids = batch
        input_ids = input_ids.to(config.device)
        target_ids = target_ids.to(config.device)
        
        loss = model.compute_loss(input_ids, target_ids)
        total_loss += loss.item()
        num_batches += 1
    
    model.train()
    return total_loss / num_batches


def save_checkpoint(
    model: LLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: TrainConfig,
    model_config: LLMConfig
):
    """Save model checkpoint."""
    os.makedirs(config.output_dir, exist_ok=True)
    
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_config": model_config.__dict__,
        "train_config": config.__dict__,
    }
    
    path = os.path.join(config.output_dir, f"checkpoint_{step}.pt")
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")
    
    # Also save as 'latest'
    latest_path = os.path.join(config.output_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    path: str,
    model: LLM,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> int:
    """
    Load model checkpoint.
    
    Returns:
        step number
    """
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint.get("step", 0)


# ============================================================
# Main Training Loop
# ============================================================

def train(
    model_config: LLMConfig,
    train_config: TrainConfig,
    tokenizer: BaseTokenizer,
    resume_from: Optional[str] = None,
    hf_train_dataset: Optional[TorchDataset] = None,
    hf_val_dataset: Optional[TorchDataset] = None,
):
    """
    Main training function.
    
    Args:
        model_config: Model hyperparameters
        train_config: Training hyperparameters
        tokenizer: Tokenizer to use
        resume_from: Path to checkpoint to resume from
    """
    print("=" * 60)
    print("Starting training")
    print("=" * 60)
    
    # Create model
    print("\nInitializing model...")
    # Ensure model vocab matches tokenizer (avoid embedding index out of range)
    if hasattr(tokenizer, "vocab_size"):
        if tokenizer.vocab_size != model_config.vocab_size:
            print(f"Adjusting model_config.vocab_size {model_config.vocab_size} -> {tokenizer.vocab_size}")
            model_config.vocab_size = tokenizer.vocab_size

    model = LLM(model_config)
    model = model.to(train_config.device)
    print(model)
    
    # Create optimizer
    # Separate weight decay for different parameter types
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't apply weight decay to biases, norms, embeddings
        if "bias" in name or "norm" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_groups = [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(
        optimizer_groups,
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2)
    )
    
    # Mixed precision scaler (for GPU training)
    # Speeds up training because 16-bit computations are faster on modern GPUs (especially NVIDIA Tensor Cores).
    # Reduces GPU memory usage, allowing larger batch sizes or models.
    scaler = None
    if train_config.device == "cuda":
        scaler = torch.cuda.amp.GradScaler()
    
    # Load checkpoint if resuming
    start_step = 0
    if resume_from:
        start_step = load_checkpoint(resume_from, model, optimizer)
        print(f"Resumed from step {start_step}")
    
    # Create datasets
    print("\nLoading datasets...")
    if hf_train_dataset is not None:
        train_dataset = HFTextDataset(hf_train_dataset, text_column="text", tokenizer=tokenizer, max_seq_len=model_config.max_seq_len)
    else:
        train_dataset = TextDataset(
            train_config.data_path,
            tokenizer,
            model_config.max_seq_len
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = None
    if train_config.val_path or hf_val_dataset is not None:
        if hf_val_dataset is not None:
            val_dataset = HFTextDataset(hf_val_dataset, text_column="text", tokenizer=tokenizer, max_seq_len=model_config.max_seq_len)
        else:
            val_dataset = TextDataset(
                train_config.val_path,
                tokenizer,
                model_config.max_seq_len
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # Training loop
    print("\nStarting training loop...")
    print(f"Total steps: {train_config.max_steps}")
    print(f"Batch size: {train_config.batch_size}")
    print(f"Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print()
    
    model.train()
    optimizer.zero_grad()
    
    step = start_step
    accumulated_loss = 0
    start_time = time.time()
    
    train_iter = iter(train_loader)
    
    while step < train_config.max_steps:
        # Get batch (restart iterator if exhausted)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Update learning rate
        lr = get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # Training step
        loss = train_step(model, batch, optimizer, train_config, scaler)
        accumulated_loss += loss
        
        # Optimizer step (after accumulation)
        if (step + 1) % train_config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            
            # Update weights
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        step += 1
        
        # Logging
        if step % train_config.log_interval == 0:
            avg_loss = accumulated_loss / train_config.log_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (
                train_config.log_interval * 
                train_config.batch_size * 
                model_config.max_seq_len / 
                elapsed
            )
            
            print(
                f"Step {step:6d} | "
                f"Loss {avg_loss:.4f} | "
                f"LR {lr:.2e} | "
                f"Tokens/s {tokens_per_sec:.0f}"
            )
            
            accumulated_loss = 0
            start_time = time.time()
        
        # Evaluation
        if val_loader and step % train_config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, train_config)
            print(f">>> Validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        if step % train_config.save_interval == 0:
            save_checkpoint(model, optimizer, step, loss, train_config, model_config)
    
    # Final save
    save_checkpoint(model, optimizer, step, loss, train_config, model_config)
    print("\nTraining complete!")


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    from config import LLMConfig
    
    # Create sample training data if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    train_file = data_dir / "train_bahubali.txt"

    # Download data if missing
    data_file = Path("data/train_tiny.txt")
    if not data_file.exists():
        print("Downloading training data...")
        import urllib.request
        import os
        
        os.makedirs("data", exist_ok=True)
        
        books = [
            "https://www.gutenberg.org/files/1342/1342-0.txt",
            "https://www.gutenberg.org/files/11/11-0.txt", 
            "https://www.gutenberg.org/files/84/84-0.txt",
            "https://www.gutenberg.org/files/1661/1661-0.txt",
            "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
            "https://www.gutenberg.org/files/1399/1399-0.txt",  # Anna Karenina
            "https://www.gutenberg.org/files/98/98-0.txt",      # Tale of Two Cities
            "https://www.gutenberg.org/files/74/74-0.txt",      # Tom Sawyer
            "https://www.gutenberg.org/files/76/76-0.txt",      # Huckleberry Finn
            "https://www.gutenberg.org/files/16/16-0.txt",      # Peter Pan
            "https://www.gutenberg.org/files/5200/5200-0.txt",  # Metamorphosis
            "https://www.gutenberg.org/files/1232/1232-0.txt",  # The Prince
            "https://www.gutenberg.org/files/2542/2542-0.txt",  # A Doll's House
            "https://www.gutenberg.org/files/174/174-0.txt",    # Dorian Gray
            "https://www.gutenberg.org/files/345/345-0.txt",    # Dracula
        ]
        
        all_text = []
        for url in books:
            name = url.split("/")[-1]
            print(f"Downloading {name}...")
            try:
                text = urllib.request.urlopen(url).read().decode('utf-8-sig')
                
                # Clean Gutenberg headers
                if "*** START OF" in text:
                    text = text.split("*** START OF", 1)[-1].split("\n", 1)[-1]
                if "*** END OF" in text:
                    text = text.split("*** END OF", 1)[0]
                
                all_text.append(text.strip())
                print(f"  ✓ {len(text):,} chars")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        combined = "\n\n".join(all_text)
        
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(combined)
        print(f"Saved to {data_file}")
    
    # Configurations
    model_config = LLMConfig(
        vocab_size=50257,  
        hidden_size=768,
        num_layers=6,
        num_heads=12,
        num_kv_heads=4,
        intermediate_size=2048,
        max_seq_len=512,
    )
    
    train_config = TrainConfig(
        data_path=str(train_file),
        batch_size=8,
        gradient_accumulation_steps=4,
        max_steps=50000,
        learning_rate=1e-4,
        warmup_steps=500,
        log_interval=50,
        save_interval=1000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Simple tokenizer for testing
    from tokenizer import TiktokenTokenizer
    tokenizer = TiktokenTokenizer("gpt2")
    
    # Train!
    # train(model_config, train_config, tokenizer)

    hf_train = load_dataset("agentlans/high-quality-english-sentences", split="train")
    hf_test = load_dataset("agentlans/high-quality-english-sentences", split="test")

    train(model_config, train_config, tokenizer, hf_train_dataset=hf_train, hf_val_dataset=hf_test)