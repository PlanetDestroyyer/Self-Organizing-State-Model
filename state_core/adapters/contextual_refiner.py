"""
Contextual MU Refinement - Phase 2.3

Adds local context awareness to position-invariant MU via a small transformer.

This is a CRITICAL DECISION POINT:
- If homonym separation >0.05: Context helps! Proceed with advanced features
- If homonym separation <0.01: Local context insufficient, skip complex approaches

Architecture:
    MU (position-invariant) → 3-token window transformer → Context-aware MU
    
    Window size: 3 tokens (i-1, i, i+1)
    Transformer: 1 layer, 4 heads
    Parameters: ~200K (minimal overhead)

Based on peer review feedback: "Test simplest context-aware approach first"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualMURefinement(nn.Module):
    """
    Refine position-invariant MU with local context.
    
    Uses a small transformer to incorporate information from neighboring tokens.
    This tests whether local context improves semantic understanding.
    """
    
    def __init__(
        self,
        mu_dim: int = 64,
        window_size: int = 3,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            mu_dim: MU state dimension (64 for 8×8 matrix)
            window_size: Context window size (default 3 = i-1, i, i+1)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Drop out rate
        """
        super().__init__()
        
        assert mu_dim % num_heads == 0, f"mu_dim ({mu_dim}) must be divisible by num_heads ({num_heads})"
        
        self.mu_dim = mu_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Transformer encoder layers for context refinement
        self.context_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=mu_dim,
                nhead=num_heads,
                dim_feedforward=mu_dim * 2,  # 2× expansion in FFN
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Pre-LayerNorm for stability
            )
            for _ in range(num_layers)
        ])
        
        # Gating mechanism: decide how much context to use
        self.context_gate = nn.Sequential(
            nn.Linear(mu_dim * 2, mu_dim),  # Concat original + refined
            nn.Sigmoid()
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Contextual MU Refinement: {total_params:,} parameters")
        print(f"    Window: {window_size} tokens, {num_layers} layers, {num_heads} heads")
    
    def forward(self, mu_state: torch.Tensor) -> torch.Tensor:
        """
        Refine MU state with local context.
        
        Args:
            mu_state: [B, T, mu_dim] position-invariant MU state
            
        Returns:
            refined: [B, T, mu_dim] context-aware MU state
        """
        B, T, D = mu_state.shape
        
        # Store original for residual/gating
        mu_original = mu_state
        
        # Process each position with its local window
        refined_states = []
        
        for i in range(T):
            # Extract window centered at position i
            start = max(0, i - self.window_size // 2)
            end = min(T, i + self.window_size // 2 + 1)
            window = mu_state[:, start:end, :]  # [B, window_len, D]
            
            # Apply transformer layers to window
            context_out = window
            for layer in self.context_layers:
                context_out = layer(context_out)  # [B, window_len, D]
            
            # Extract refined state for center token
            center_idx = i - start
            refined_token = context_out[:, center_idx, :]  # [B, D]
            
            refined_states.append(refined_token)
        
        # Stack refined states
        refined = torch.stack(refined_states, dim=1)  # [B, T, D]
        
        # Gated fusion: learn to balance original vs. context-refined
        combined = torch.cat([mu_original, refined], dim=-1)  # [B, T, 2D]
        gate = self.context_gate(combined)  # [B, T, D]
        
        # Final output: weighted combination
        output = gate * refined + (1 - gate) * mu_original
        
        return output


class ContextualMURefinementEfficient(nn.Module):
    """
    More efficient version using causal convolution instead of sliding windows.
    
    Uses 1D conv with kernel_size=3 to capture local context in one pass.
    Faster than the sliding window approach above.
    """
    
    def __init__(
        self,
        mu_dim: int = 64,
        kernel_size: int = 3,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.mu_dim = mu_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        # 1D conv layers for local context
        self.context_conv = nn.ModuleList([
            nn.Conv1d(
                in_channels=mu_dim,
                out_channels=mu_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # Same padding
                groups=mu_dim  # Depthwise conv for efficiency
            )
            for _ in range(num_layers)
        ])
        
        # Pointwise conv (1×1) for mixing channels
        self.pointwise = nn.ModuleList([
            nn.Conv1d(mu_dim, mu_dim, kernel_size=1)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(mu_dim)
            for _ in range(num_layers)
        ])
        
        # Gating
        self.context_gate = nn.Sequential(
            nn.Linear(mu_dim * 2, mu_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Contextual MU Refinement (Efficient): {total_params:,} parameters")
        print(f"    Kernel: {kernel_size}, {num_layers} layers")
    
    def forward(self, mu_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu_state: [B, T, mu_dim]
            
        Returns:
            refined: [B, T, mu_dim]
        """
        B, T, D = mu_state.shape
        
        # Store original
        mu_original = mu_state
        
        # Conv expects [B, C, T]
        x = mu_state.transpose(1, 2)  # [B, D, T]
        
        # Apply conv layers
        for conv, pointwise, norm in zip(self.context_conv, self.pointwise, self.norms):
            # Depthwise conv
            x_conv = conv(x)
            
            # Pointwise conv
            x_point = pointwise(x_conv)
            
            # Residual + norm
            x = x + self.dropout(x_point)
            x = norm(x.transpose(1, 2)).transpose(1, 2)  # Norm in [B, T, D]
        
        # Back to [B, T, D]
        refined = x.transpose(1, 2)
        
        # Gated fusion
        combined = torch.cat([mu_original, refined], dim=-1)
        gate = self.context_gate(combined)
        
        output = gate * refined + (1 - gate) * mu_original
        
        return output


# ============================================================================
# Testing
# ============================================================================

def test_contextual_refinement():
    """Test contextual MU refinement."""
    print("Testing Contextual MU Refinement...")
    
    # Test transformer-based version
    refiner = ContextualMURefinement(
        mu_dim=64,
        window_size=3,
        num_heads=4,
        num_layers=1
    )
    
    # Test forward
    B, T, D = 2, 10, 64
    mu_state = torch.randn(B, T, D)
    
    refined = refiner(mu_state)
    
    assert refined.shape == (B, T, D), f"Expected {(B, T, D)}, got {refined.shape}"
    print(f"✓ Transformer version: {refined.shape}")
    
    # Test efficient version
    refiner_eff = ContextualMURefinementEfficient(
        mu_dim=64,
        kernel_size=3,
        num_layers=1
    )
    
    refined_eff = refiner_eff(mu_state)
    assert refined_eff.shape == (B, T, D)
    print(f"✓ Efficient (conv) version: {refined_eff.shape}")
    
    # Test that it actually changes the state
    change = (refined - mu_state).abs().mean().item()
    print(f"✓ Average change from original: {change:.6f}")
    
    print("\n✅ All contextual refinement tests passed!")


if __name__ == '__main__':
    test_contextual_refinement()
