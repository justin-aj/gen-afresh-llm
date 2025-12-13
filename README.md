Modern LLMs (LLaMA style) use Pre-Norm:

    x ──────────────────────────┐
    │                           │
    ▼                           │
  RMSNorm                       │
    │                           │
    ▼                           │
  Attention                     │
    │                           │
    ▼                           │
    + ◄─────────────────────────┘  (residual connection)
    │
    ──────────────────────────┐
    │                         │
    ▼                         │
  RMSNorm                     │
    │                         │
    ▼                         │
  SwiGLU MLP                  │
    │                         │
    ▼                         │
    + ◄───────────────────────┘  (residual connection)
    │
    ▼
  output



```

---

## **Visual flow:**
```
Layer 0:
    x ──→ [Norm → Attn → +] ──→ [Norm → MLP → +] ──→ output₀
                    ↑                      ↑
                 residual              residual

Layer 1:
    output₀ ──→ [Norm → Attn → +] ──→ [Norm → MLP → +] ──→ output₁

    ...

Layer N:
    outputₙ₋₁ ──→ [Norm → Attn → +] ──→ [Norm → MLP → +] ──→ final output
```

---

## **Why Pre-Norm instead of Post-Norm?**
```
Post-Norm (original Transformer, GPT-2):
    x → Attention → + → Norm → MLP → + → Norm
    
    Problem: Gradients can explode/vanish in deep networks

Pre-Norm (LLaMA, modern LLMs):
    x → Norm → Attention → + → Norm → MLP → +
    
    Benefit: More stable gradients, easier to train deep models


```

---

## **Visual architecture:**
```
Input tokens: [The, cat, sat, on, the, mat]
                │
                ▼
        ┌───────────────┐
        │   Embedding   │  (vocab_size → hidden_size)
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │  Decoder × N  │  (self-attention + MLP)
        │    Layer 0    │
        │    Layer 1    │
        │     ...       │
        │    Layer N    │
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │   RMSNorm     │
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │  Output Proj  │  (hidden_size → vocab_size)
        └───────────────┘
                │
                ▼
Output logits: [prob distribution over vocabulary]
                     → sample "purred"


## **How BPE tokenization works:**
```
Training (done once, creates the vocabulary):
    1. Start with characters: ["h", "e", "l", "l", "o"]
    2. Find most common pair: ("l", "l") → merge to "ll"
    3. Find next most common: ("he", "llo") → merge
    4. Repeat until vocab_size reached

Encoding "hello world":
    "hello world"
    → ["hel", "lo", " ", "world"]   (subwords)
    → [1542, 831, 220, 1917]        (token IDs)

This handles:
    ✓ Unknown words (break into known subwords)
    ✓ Efficient (common words = single token)
    ✓ Multilingual (just characters if needed)
```

---

## **Visual comparison:**
```
Character tokenizer:
    "Hello" → ["H", "e", "l", "l", "o"] → 5 tokens

Word tokenizer:
    "Hello" → ["Hello"] → 1 token
    "Helloo" → ["<UNK>"] → can't handle!

BPE tokenizer:
    "Hello" → ["Hello"] → 1 token
    "Helloo" → ["Hell", "oo"] → 2 tokens ✓


```

---

## **What each utility does:**

| Function | Purpose |
|----------|---------|
| `set_seed()` | Reproducible experiments |
| `get_device()` | Auto-detect best device |
| `count_parameters()` | Count model params |
| `estimate_memory()` | Estimate GPU memory needed |
| `print_model_summary()` | Nice model overview |
| `load_pretrained_weights()` | Load LLaMA/other weights |
| `get_llama_key_mapping()` | Map LLaMA weight names |
| `TrainingLogger` | Track & save metrics |
| `save_config/load_config()` | Config management |
| `benchmark_model()` | Measure speed |

---