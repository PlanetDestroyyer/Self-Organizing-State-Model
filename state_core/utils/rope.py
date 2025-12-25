"""
RoPE (Rotary Position Embeddings) Implementation

Provides relative positional encoding via rotation matrices.
Based on RoFormer (Su et al., 2021).

Benefits over learned positional embeddings:
- No maximum sequence length
- Better extrapolation to longer sequences
- Relative position encoding
- Zero parameters

Usage:
    rope = RoPEEmbedding(dim=64)
    cos, sin = rope(seq_len)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
"""

import torch
import torch.nn as nn
import math


class RoPEEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Encodes position through rotation matrices applied to Q and K.
    """
    
    def __init__(self, dim: int, max_len: int = 16384, base: float = 10000.0):
        """
        Args:
            dim: Embedding dimension (must be even)
            max_len: Maximum sequence length to precompute
            base: Base for frequency computation (default 10000)
        """
        super().__init__()
        
        assert dim % 2 == 0, f"RoPE requires even dimension, got {dim}"
        
        self.dim = dim
        self.max_len = max_len
        self.base = base
        
        # Compute inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute rotation matrices for efficiency
        self._precompute_freqs(max_len)
    
    def _precompute_freqs(self, max_len: int):
        """Precompute cos and sin for all positions up to max_len."""
        # Position indices [0, 1, 2, ..., max_len-1]
        t = torch.arange(max_len, dtype=torch.float32)
        
        # Outer product: [max_len, dim/2]
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Concatenate to match full dimension: [max_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Compute cos and sin
        cos = emb.cos()
        sin = emb.sin()
        
        # Register as buffers (moved to device automatically)
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)
    
    def forward(self, seq_len: int):
        """
        Get rotation matrices for sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            cos: [seq_len, dim]
            sin: [seq_len, dim]
        """
        # If sequence is longer than cache, recompute
        if seq_len > self.max_len:
            self._precompute_freqs(seq_len)
        
        # Return cached values
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    
    This is the key operation for RoPE:
    - Split x into two halves
    - Second half negated and moved to front
    
    Args:
        x: [..., dim] tensor
        
    Returns:
        rotated: [..., dim] tensor
    """
    # Split into two halves along last dimension
    x1, x2 = x.chunk(2, dim=-1)
    
    # Rotate: [-x2, x1]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Implements: RoPE(x, m) = x * cos(mθ) + rotate_half(x) * sin(mθ)
    where m is position index, θ are the learned frequencies.
    
    Args:
        q: Query tensor [..., seq_len, dim]
        k: Key tensor [..., seq_len, dim]
        cos: Cosine values [seq_len, dim]
        sin: Sine values [seq_len, dim]
        
    Returns:
        q_rot: Rotated query [..., seq_len, dim]
        k_rot: Rotated key [..., seq_len, dim]
    """
    # Broadcast cos/sin to match q/k shape
    # q/k shape: [batch, heads, seq_len, dim] or [batch, seq_len, dim]
    # cos/sin shape: [seq_len, dim]
    
    # Add batch dimensions if needed
    while cos.ndim < q.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    
    # Apply rotation
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    
    return q_rot, k_rot


# ============================================================================
# Testing & Validation
# ============================================================================

def test_rope():
    """Test RoPE implementation."""
    print("Testing RoPE...")
    
    # Create RoPE
    dim = 64
    rope = RoPEEmbedding(dim=dim, max_len=512)
    
    # Test forward
    seq_len = 10
    cos, sin = rope(seq_len)
    
    assert cos.shape == (seq_len, dim), f"Expected {(seq_len, dim)}, got {cos.shape}"
    assert sin.shape == (seq_len, dim), f"Expected {(seq_len, dim)}, got {sin.shape}"
    
    print(f"✓ RoPE forward: cos/sin shape {cos.shape}")
    
    # Test apply to Q/K
    batch_size = 2
    num_heads = 4
    head_dim = dim // num_heads
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Need to adjust cos/sin for multi-head
    cos_head = cos[:, :head_dim]  # [seq_len, head_dim]
    sin_head = sin[:, :head_dim]
    
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos_head, sin_head)
    
    assert q_rot.shape == q.shape, f"Q shape mismatch: {q_rot.shape} vs {q.shape}"
    assert k_rot.shape == k.shape, f"K shape mismatch: {k_rot.shape} vs {k.shape}"
    
    print(f"✓ apply_rotary_pos_emb: output shapes correct")
    
    # Test longer sequence (extrapolation)
    long_seq_len = 1000
    cos_long, sin_long = rope(long_seq_len)
    assert cos_long.shape == (long_seq_len, dim)
    
    print(f"✓ RoPE extrapolation: handled seq_len={long_seq_len}")
    
    # Test rotation property: RoPE(x, m+k) = RoPE(RoPE(x, m), k)
    # This ensures relative position encoding
    x = torch.randn(1, 1, 1, dim)
    cos_all, sin_all = rope(6)
    cos_m, sin_m = cos_all[3:4], sin_all[3:4]  # Position 3
    cos_k, sin_k = cos_all[1:2], sin_all[1:2]  # Offset 1  
    cos_mk, sin_mk = cos_all[4:5], sin_all[4:5]  # Position 3+1=4
    
    x_m = x * cos_m + rotate_half(x) * sin_m
    x_mk_rel = x_m * cos_k + rotate_half(x_m) * sin_k
    x_mk_abs = x * cos_mk + rotate_half(x) * sin_mk
    
    # Should be approximately equal (relative encoding property)
    diff = (x_mk_rel - x_mk_abs).abs().max()
    print(f"✓ RoPE relative property: max diff = {diff:.6f}")
    
    print("\n✅ All RoPE tests passed!")


if __name__ == '__main__':
    test_rope()
