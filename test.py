# debug.py
import torch
from config import LLMConfig
from model import LLM
from tokenizer import TiktokenTokenizer

# Load checkpoint
checkpoint = torch.load("checkpoints/checkpoint_latest.pt", map_location="cpu")

print("="*60)
print("CHECKPOINT INFO")
print("="*60)
print(f"Training step: {checkpoint.get('step', 'unknown')}")
print(f"Loss at save: {checkpoint.get('loss', 'unknown')}")
print(f"Model vocab_size: {checkpoint['model_config']['vocab_size']}")

# Load tokenizer
tokenizer = TiktokenTokenizer("cl100k_base")
print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

# Check match
vocab_match = checkpoint['model_config']['vocab_size'] == tokenizer.vocab_size
print(f"Vocab match: {vocab_match}")

if not vocab_match:
    print("\n❌ VOCAB MISMATCH - This is your problem!")
    print(f"   Model trained with: {checkpoint['model_config']['vocab_size']}")
    print(f"   Tokenizer has:      {tokenizer.vocab_size}")
    exit()

# Load model
config = LLMConfig(**checkpoint['model_config'])
model = LLM(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test
print("\n" + "="*60)
print("GENERATION TEST")
print("="*60)

prompt = "The"
tokens = tokenizer.encode(prompt)
print(f"Prompt: '{prompt}' → tokens: {tokens}")

input_ids = torch.tensor([tokens])

with torch.no_grad():
    output = model(input_ids)
    logits = output["logits"][0, -1, :]
    
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 10)
    
    print("\nTop 10 next token predictions:")
    for prob, idx in zip(top_probs, top_indices):
        token_str = tokenizer.decode([idx.item()])
        print(f"  {idx.item():6d} | prob {prob.item():.4f} | '{token_str}'")

print(f"\nLogits stats:")
print(f"  Std:  {logits.std().item():.4f}")
print(f"  Max:  {logits.max().item():.4f}")
print(f"  Min:  {logits.min().item():.4f}")