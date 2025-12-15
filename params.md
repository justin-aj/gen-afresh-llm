# train.py - Data processing
class TextDataset:
    def __init__(self, file_path, tokenizer, max_seq_len, stride=None):
        self.max_seq_len = max_seq_len   # ← Training context
        self.stride = stride or max_seq_len // 2  # ← Data overlap
```

---

## **5. Architecture Choices (across files)**

These are structural decisions baked into the code:

| Choice | Location | Alternatives | Impact |
|--------|----------|--------------|--------|
| **RMSNorm vs LayerNorm** | `normalization.py` | LayerNorm | RMSNorm is faster, equally effective |
| **Pre-norm vs Post-norm** | `decoder.py` | Post-norm | Pre-norm trains more stably |
| **SwiGLU vs ReLU MLP** | `mlp.py` | GELU, ReLU | SwiGLU performs better empirically |
| **RoPE vs Absolute Position** | `positional.py` | Learned, Sinusoidal | RoPE extrapolates better |
| **GQA vs MHA** | `attention.py` | Full MHA | GQA is faster with minimal quality loss |
| **Tied embeddings** | `model.py` | Untied | Tied saves parameters |

---

## **6. Complete Hyperparameter Map**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PERFORMANCE FACTORS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   ARCHITECTURE  │  │    TRAINING     │  │   GENERATION    │             │
│  │   (config.py)   │  │   (train.py)    │  │  (generate.py)  │             │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤             │
│  │ vocab_size      │  │ learning_rate ⭐│  │ temperature ⭐  │             │
│  │ hidden_size ⭐  │  │ batch_size      │  │ top_k           │             │
│  │ num_layers ⭐   │  │ grad_accum      │  │ top_p           │             │
│  │ num_heads       │  │ max_steps       │  │ repetition_pen  │             │
│  │ num_kv_heads    │  │ warmup_steps    │  │ max_new_tokens  │             │
│  │ intermediate    │  │ weight_decay    │  │                 │             │
│  │ max_seq_len     │  │ grad_clip       │  │                 │             │
│  │ rope_theta      │  │ beta1, beta2    │  │                 │             │
│  │ dropout         │  │ min_lr          │  │                 │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │      DATA       │  │ ARCH CHOICES    │  │   NUMERICAL     │             │
│  │   (train.py)    │  │  (all files)    │  │   (implicit)    │             │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤             │
│  │ data quality ⭐ │  │ RMSNorm         │  │ fp16 vs fp32    │             │
│  │ data size ⭐    │  │ Pre-norm        │  │ attention scale │             │
│  │ seq_len         │  │ SwiGLU          │  │ init std (0.02) │             │
│  │ stride          │  │ RoPE            │  │ eps (1e-6)      │             │
│  │ tokenizer       │  │ GQA             │  │                 │             │
│  │                 │  │ tied embeddings │  │                 │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
│  ⭐ = Most impactful                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## **7. Scaling Laws (what to increase for better performance)**

Based on research (Chinchilla, etc.), here's what matters most:
```
Performance ∝ (Parameters)^0.5 × (Data)^0.5 × (Compute)^0.5

┌────────────────────────────────────────────────────────────────┐
│                     SCALING PRIORITY                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. DATA (most important)                                      │
│     └── More high-quality text beats bigger models            │
│                                                                │
│  2. PARAMETERS (model size)                                    │
│     └── hidden_size × num_layers                              │
│     └── Double params ≈ 15% better                            │
│                                                                │
│  3. COMPUTE (training steps × batch size)                      │
│     └── Train longer with more data                           │
│                                                                │
│  4. ARCHITECTURE (diminishing returns)                         │
│     └── GQA, SwiGLU, RoPE are already optimal choices        │
│                                                                │
└────────────────────────────────────────────────────────────────┘