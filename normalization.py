import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    Unlike LayerNorm, RMSNorm doesn't center the activations (no mean subtraction).
    It only rescales by the root mean square, making it simpler and faster.
    
    Formula: output = (x / RMS(x)) * weight
    Where RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Args:
            hidden_size: Dimension of the input features
            eps: Small constant for numerical stability (avoid division by zero)
        """
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter (one per feature)
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
        
        Returns:
            Normalized tensor of same shape
        """
        # Step 1: Compute mean of squares along last dimension
        # x^2 → mean across hidden_size → shape: (batch, seq_len, 1)
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)

        # Basically for each word, we get the mean, Mean of 768 values
        
        # Step 2: Compute RMS (add eps for stability)
        rms = torch.sqrt(mean_square + self.eps)

        # For each 768, we get one rms finally
                
        # Step 3: Normalize and scale by learnable weight
        x_normalized = x / rms

        # Those 768 values / rms of that 768 values for 10 sequence
        
        return x_normalized * self.weight


if __name__ == "__main__":
    # Test RMSNorm
    batch_size, seq_len, hidden_size = 2, 10, 768
    
    x = torch.randn(batch_size, seq_len, hidden_size)

    #   |-----------------------------------------------------
    #   | The width is 10, the length of each is 768, in 2 batches
    #   |
    #   |
    #   |
    #   |
    #   |   
    #   |
    #   |   
    #   |-----------------------------------------------------

    norm = RMSNorm(hidden_size)
    
    output = norm(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weight shape: {norm.weight.shape}")
    
    # Verify RMS of output is approximately 1
    rms_out = output.pow(2).mean(dim=-1).sqrt().mean()
    print(f"Output RMS (should be ~1): {rms_out:.4f}")
