"""
PairNorm: Normalization layer for preventing oversmoothing in Graph Neural Networks

Based on:
    "PairNorm: Tackling Oversmoothing in GNNs" (Zhao & Akoglu, 2020)
    ICLR 2020
    https://openreview.net/forum?id=rkecl1rtwB

PairNorm addresses the oversmoothing problem in deep GNNs where node
representations become indistinguishable after multiple message-passing layers.
This is critical for SOSM's graph-based attention mechanism.

Mathematical Formulation:
    1. Centering: x̃ = x - mean(x)
    2. Scaling: x̂ = s · x̃ / sqrt(mean(||x̃||²_2))

where s is a learned or fixed scaling parameter.
"""

import torch
import torch.nn as nn


class PairNorm(nn.Module):
    """
    PairNorm layer for preventing graph over-smoothing.
    
    This normalization maintains constant pairwise distances between node
    representations across layers, preventing the collapse to a single point
    (over-smoothing) that naturally occurs in graph neural networks.
    
    In SOSM context, it prevents token representations from becoming too
    similar after multiple rounds of graph-based attention.
    
    Args:
        scale: Scaling factor (default: 1.0)
            If learnable=True, this is the initial value
        scale_individually: Whether to maintain scale per node or globally
            (default: False, recommended for most cases)
        learnable: Whether the scale parameter should be learned (default: False)
        eps: Small constant for numerical stability (default: 1e-5)
    
    Shape:
        Input: (N, D) where N = number of nodes, D = feature dimension
               or (B, N, D) for batched graphs
        Output: Same shape as input
    
    Example:
        >>> norm = PairNorm(scale=1.0)
        >>> x = torch.randn(100, 64)  # 100 nodes, 64 features
        >>> x_normalized = norm(x)
    """
    
    def __init__(self, scale=1.0, scale_individually=False, learnable=False, eps=1e-5):
        super().__init__()
        self.scale_individually = scale_individually
        self.eps = eps
        
        if learnable:
            self.scale = nn.Parameter(torch.tensor(scale))
        else:
            self.register_buffer('scale', torch.tensor(scale))
    
    def forward(self, x):
        """
        Apply PairNorm normalization.
        
        Args:
            x: Input tensor of shape (N, D) or (B, N, D)
        
        Returns:
            Normalized tensor of same shape as input
        """
        # Handle both (N, D) and (B, N, D) shapes
        original_shape = x.shape
        if x.dim() == 3:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
        elif x.dim() == 2:
            N, D = x.shape
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
        
        # Step 1: Centering - subtract mean
        # Mean across all nodes: [D]
        x_mean = x.mean(dim=0, keepdim=True)  # [1, D]
        x_centered = x - x_mean  # [N, D]
        
        # Step 2: Scaling
        if self.scale_individually:
            # Scale each dimension independently
            # Norm per dimension: [D]
            norm_per_dim = torch.sqrt((x_centered ** 2).mean(dim=0) + self.eps)  # [D]
            x_normalized = self.scale * x_centered / norm_per_dim.unsqueeze(0)  # [N, D]
        else:
            # Global scaling (recommended)
            # Mean squared norm across all nodes and dimensions
            mean_norm = torch.sqrt((x_centered ** 2).mean() + self.eps)  # scalar
            x_normalized = self.scale * x_centered / mean_norm  # [N, D]
        
        # Reshape back to original
        if len(original_shape) == 3:
            x_normalized = x_normalized.reshape(original_shape)
        
        return x_normalized


class PairNormBatched(nn.Module):
    """
    Variant of PairNorm that normalizes within each batch element separately.
    
    This is useful when different samples in a batch should be normalized
    independently (e.g., different graphs in a batch).
    
    Args:
        scale: Scaling factor (default: 1.0)
        learnable: Whether the scale parameter should be learned (default: False)
        eps: Small constant for numerical stability (default: 1e-5)
    """
    
    def __init__(self, scale=1.0, learnable=False, eps=1e-5):
        super().__init__()
        self.eps = eps
        
        if learnable:
            self.scale = nn.Parameter(torch.tensor(scale))
        else:
            self.register_buffer('scale', torch.tensor(scale))
    
    def forward(self, x):
        """
        Apply PairNorm normalization per batch element.
        
        Args:
            x: Input tensor of shape (B, N, D)
                B = batch size
                N = number of nodes/tokens
                D = feature dimension
        
        Returns:
            Normalized tensor of shape (B, N, D)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, N, D), got {x.dim()}D")
        
        B, N, D = x.shape
        
        # Center per batch element
        x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        x_centered = x - x_mean  # [B, N, D]
        
        # Compute norm per batch element
        # Mean squared norm per batch: [B]
        mean_norm = torch.sqrt((x_centered ** 2).mean(dim=[1, 2], keepdim=True) + self.eps)  # [B, 1, 1]
        
        # Scale
        x_normalized = self.scale * x_centered / mean_norm  # [B, N, D]
        
        return x_normalized
