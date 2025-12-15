# Step 1: config.py

from dataclasses import dataclass

@dataclass
class LLMConfig:

    # Determines the size of the embedding matrix, shape [vocab_size, hidden_size]
    # Large vocab_size can capture more nuance but increases model size
    vocab_size: int = 100277 # Number of unique words

    # Dim of token embeddings and hidden states
    # Internal computations operate on vectors of this size 
    hidden_size: int = 768 # 

    # Number of transformer blocks
    # Each layer has self-attention + FFN
    # More Layers, capture deeper complex patterns, increase compute cost
    num_layers: int = 12

    # Number of attention heads in Multi-Head attention
    # MHA splits hidden_size into num_heads, head_dim = hidden_size / num_heads
    num_heads: int = 12

    # Number of key/value heads used for Grouped Query Attention (GQA)
    num_kv_heads: int = 4        # for Grouped Query Attention

    # Size of the hidden layer inside the FFN
    intermediate_size: int = 2048  # MLP hidden dim

    # Maximum sequence length the model can handle.
    max_seq_len: int = 2048

    # randomly sets some activations to zero to prevent overfitting
    dropout: float = 0.0

    # Scaling factor for Rotary Positional Embeddings (RoPE)
    rope_theta: float = 10000.0
    
    tie_embeddings: bool = True