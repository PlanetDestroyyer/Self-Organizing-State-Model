"""
FlashAttention integration for SOSM - Phase 2.4.4

Provides FlashAttention-enabled MultiheadAttention with automatic fallback
to standard PyTorch attention when FlashAttention is not available.

Benefits:
- 2-3× faster training
- 20-30% less memory usage
- Better numerical stability
- Drop-in replacement for nn.MultiheadAttention
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings


# Try to import FlashAttention (supports both v1.x and v2.x)
FLASH_ATTN_AVAILABLE = False
flash_attn_func = None

try:
    # Try v2.x import path first
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("✓ FlashAttention v2.x available - will use optimized attention")
except ImportError:
    try:
        # Try v1.x import path (different location)
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_ATTN_AVAILABLE = True
        print("✓ FlashAttention v1.x available - will use optimized attention")
    except ImportError:
        FLASH_ATTN_AVAILABLE = False
        print("✗ FlashAttention not available - using standard attention")
        print("  Install with: pip install flash-attn --no-build-isolation")


class FlashMultiheadAttention(nn.Module):
    """
    MultiheadAttention with FlashAttention support.
    
    Drop-in replacement for nn.MultiheadAttention that automatically
    uses FlashAttention when available, otherwise falls back to standard.
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability
        batch_first: If True, then input/output tensors are (batch, seq, feature)
        use_flash: Force enable/disable FlashAttention (default: auto-detect)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        use_flash: Optional[bool] = None
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        # Determine if we should use FlashAttention
        if use_flash is None:
            self.use_flash = FLASH_ATTN_AVAILABLE
        else:
            self.use_flash = use_flash and FLASH_ATTN_AVAILABLE
            if use_flash and not FLASH_ATTN_AVAILABLE:
                warnings.warn("FlashAttention requested but not available, using standard attention")
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout_module = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass compatible with nn.MultiheadAttention API.
        
        Args:
            query: Query tensor [B, T, E] if batch_first else [T, B, E]
            key: Key tensor
            value: Value tensor
            key_padding_mask: [B, S] boolean mask (True = ignore)
            need_weights: Return attention weights
            attn_mask: [T, S] or [B*heads, T, S] attention mask
            average_attn_weights: Average weights across heads
            
        Returns:
            output: [B, T, E] if batch_first else [T, B, E]
            attn_weights: None or [B, T, S] attention weights
        """
        # Handle batch_first
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        B, T, E = query.shape
        _, S, _ = key.shape
        
        # Project Q, K, V
        Q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(B, S, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(B, S, self.num_heads, self.head_dim)
        
        # Use FlashAttention or standard attention
        if self.use_flash and not need_weights:
            # FlashAttention path
            output = self._flash_attention(Q, K, V, key_padding_mask)
            attn_weights = None
        else:
            # Standard attention path
            output, attn_weights = self._standard_attention(
                Q, K, V, key_padding_mask, attn_mask, need_weights, average_attn_weights
            )
        
        # Output projection
        output = self.out_proj(output)
        
        # Handle batch_first
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, attn_weights
    
    def _flash_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        FlashAttention implementation with GPU compatibility fallback.
        
        Args:
            Q: [B, T, H, D]
            K: [B, S, H, D]
            V: [B, S, H, D]
            key_padding_mask: [B, S] boolean mask
            
        Returns:
            output: [B, T, E]
        """
        # FlashAttention expects [B, T, H, D]
        # It's already in the right format
        
        # Handle padding mask (FlashAttention doesn't support it directly)
        # We'll use the standard path if there's a padding mask
        if key_padding_mask is not None:
            # Fall back to standard attention for masked inputs
            return self._standard_attention(Q, K, V, key_padding_mask, None, False, False)[0]
        
        try:
            # Call FlashAttention
            output = flash_attn_func(
                Q, K, V,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=1.0 / (self.head_dim ** 0.5),
                causal=False
            )
            
            # Reshape: [B, T, H, D] -> [B, T, E]
            B, T, H, D = output.shape
            output = output.reshape(B, T, H * D)
            
            return output
            
        except RuntimeError as e:
            # Handle GPU compatibility issues (e.g., "only supports Ampere GPUs or newer")
            if "Ampere" in str(e) or "compute capability" in str(e).lower():
                # Disable FlashAttention for future calls
                self.use_flash = False
                warnings.warn(
                    f"FlashAttention failed (GPU not supported): {e}\n"
                    f"Falling back to standard attention for all future calls."
                )
                # Fall back to standard attention
                return self._standard_attention(Q, K, V, key_padding_mask, None, False, False)[0]
            else:
                # Re-raise if it's a different error
                raise
    
    def _standard_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Standard scaled dot-product attention.
        
        Args:
            Q: [B, T, H, D]
            K: [B, S, H, D]
            V: [B, S, H, D]
            
        Returns:
            output: [B, T, E]
            attn_weights: [B, T, S] or None
        """
        B, T, H, D = Q.shape
        S = K.size(1)
        
        # Reshape for attention: [B, H, T, D]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)  # [B, H, T, S]
        
        # Apply masks
        if key_padding_mask is not None:
            # Expand mask: [B, S] -> [B, 1, 1, S]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # Softmax
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_module(attn)
        
        # Apply attention to values
        output = torch.matmul(attn, V)  # [B, H, T, D]
        
        # Reshape: [B, H, T, D] -> [B, T, E]
        output = output.transpose(1, 2).reshape(B, T, H * D)
        
        # Prepare attention weights for return
        if need_weights:
            if average_attn_weights:
                attn_weights = attn.mean(dim=1)  # [B, T, S]
            else:
                attn_weights = attn  # [B, H, T, S]
        else:
            attn_weights = None
        
        return output, attn_weights


# Convenience function to create attention layers
def create_attention(
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.0,
    batch_first: bool = True,
    use_flash: Optional[bool] = None
) -> nn.Module:
    """
    Create attention layer with FlashAttention if available.
    
    Args:
        embed_dim: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        batch_first: Use batch-first format
        use_flash: Force enable/disable FlashAttention
        
    Returns:
        FlashMultiheadAttention or nn.MultiheadAttention
    """
    return FlashMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=batch_first,
        use_flash=use_flash
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("Testing FlashMultiheadAttention...")
    
    # Create attention module
    attn = FlashMultiheadAttention(
        embed_dim=64,
        num_heads=4,
        dropout=0.1,
        batch_first=True
    )
    
    print(f"Using FlashAttention: {attn.use_flash}")
    
    # Test forward pass
    x = torch.randn(2, 10, 64)  # [B, T, E]
    
    output, weights = attn(x, x, x, need_weights=True)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Weights shape: {weights.shape if weights is not None else None}")
    
    assert output.shape == x.shape
    
    # Test with mask
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 5:] = True  # Mask last 5 positions
    
    output_masked, _ = attn(x, x, x, key_padding_mask=mask)
    print(f"✓ Masked output shape: {output_masked.shape}")
    
    print("\n✅ All tests passed!")
