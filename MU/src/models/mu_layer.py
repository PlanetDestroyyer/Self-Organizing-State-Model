"""
Meaning Unit (MU) Attention Layer

This module implements the core MU attention mechanism that operates on
MU matrices instead of traditional dense embeddings.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MUAttentionLayer(nn.Module):
    """
    MU Attention Layer with semantic-role-aware gating

    This layer processes sequences of MU matrices [B, T, r, c] and applies
    multi-head attention with position-specific sensitivity gating.

    Args:
        r: Number of rows in MU matrix
        c: Number of columns in MU matrix
        d_model: Hidden dimension for attention computation
        n_heads: Number of attention heads
        dropout: Dropout probability
        sensitivity_mask: [r, c] tensor specifying update sensitivity for each position
    """

    def __init__(
        self,
        r: int = 4,
        c: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        sensitivity_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.r = r
        self.c = c
        self.rc = r * c
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Projection matrices for Q, K, V
        self.W_q = nn.Linear(self.rc, d_model)
        self.W_k = nn.Linear(self.rc, d_model)
        self.W_v = nn.Linear(self.rc, d_model)

        # Output projection
        self.W_out = nn.Linear(d_model, self.rc)

        # Gate projection for adaptive updates
        self.W_g = nn.Linear(d_model, self.rc)

        # Bias projection
        self.W_b = nn.Linear(d_model, self.rc)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm([r, c], eps=1e-6)

        # Sensitivity mask (controls how much each position can change)
        if sensitivity_mask is None:
            # Default sensitivity mask based on semantic roles
            sensitivity_mask = torch.tensor([
                [0.1, 0.01, 0.01, 0.7],   # Identity (low), Invariants (very low), Relation (high)
                [0.7, 0.7, 0.7, 0.9],      # Relations (high), Context start (very high)
                [0.9, 0.9, 0.9, 0.6],      # Context (very high), Transform (medium-high)
                [0.6, 0.5, 0.5, 0.1]       # Transform (medium-high), Compositional (medium), Global (low)
            ], dtype=torch.float32)

        self.register_buffer('sensitivity_mask', sensitivity_mask)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_out, self.W_g, self.W_b]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split hidden dimension into multiple heads

        Args:
            x: [B, T, d_model]

        Returns:
            [B, n_heads, T, d_head]
        """
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.transpose(1, 2)  # [B, n_heads, T, d_head]

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge attention heads back into single dimension

        Args:
            x: [B, n_heads, T, d_head]

        Returns:
            [B, T, d_model]
        """
        B, _, T, _ = x.shape
        x = x.transpose(1, 2)  # [B, T, n_heads, d_head]
        return x.contiguous().view(B, T, self.d_model)

    def forward(
        self,
        M: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of MU attention layer

        Args:
            M: Input MU sequence [B, T, r, c]
            mask: Optional attention mask [B, T] or [B, T, T]

        Returns:
            M_updated: Updated MU sequence [B, T, r, c]
        """
        B, T, r, c = M.shape
        assert r == self.r and c == self.c, f"Expected MU shape [..., {self.r}, {self.c}], got [..., {r}, {c}]"

        # Step 1: Flatten MUs to vectors
        M_flat = M.view(B, T, self.rc)  # [B, T, 16]

        # Step 2: Project to Q, K, V
        Q = self.W_q(M_flat)  # [B, T, d_model]
        K = self.W_k(M_flat)  # [B, T, d_model]
        V = self.W_v(M_flat)  # [B, T, d_model]

        # Step 3: Split into multiple heads
        Q_heads = self.split_heads(Q)  # [B, n_heads, T, d_head]
        K_heads = self.split_heads(K)  # [B, n_heads, T, d_head]
        V_heads = self.split_heads(V)  # [B, n_heads, T, d_head]

        # Step 4: Compute attention scores
        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1))  # [B, n_heads, T, T]
        scores = scores / math.sqrt(self.d_head)

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # Expand mask to [B, 1, 1, T] for broadcasting
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # Expand mask to [B, 1, T, T] for broadcasting
                mask = mask.unsqueeze(1)

            scores = scores.masked_fill(~mask, float('-inf'))

        # Step 5: Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, n_heads, T, T]
        attn_weights = self.dropout_layer(attn_weights)

        # Step 6: Apply attention to values
        context = torch.matmul(attn_weights, V_heads)  # [B, n_heads, T, d_head]
        context = self.merge_heads(context)  # [B, T, d_model]

        # Step 7: Project back to MU space
        delta_flat = self.W_out(context)  # [B, T, rc]
        delta_M = delta_flat.view(B, T, r, c)  # [B, T, r, c]

        # Step 8: Compute adaptive gates
        G_flat = torch.sigmoid(self.W_g(context))  # [B, T, rc]
        G = G_flat.view(B, T, r, c)  # [B, T, r, c]

        # Apply sensitivity mask to gates
        G = G * self.sensitivity_mask.unsqueeze(0).unsqueeze(0)  # [B, T, r, c]

        # Step 9: Compute bias term
        B_flat = torch.tanh(self.W_b(context))  # [B, T, rc]
        B_term = B_flat.view(B, T, r, c)  # [B, T, r, c]

        # Step 10: Update MU with gated residual connection
        M_updated = M * (1 - G) + delta_M * G + B_term * 0.1  # Small bias scale

        # Step 11: Layer normalization
        M_updated = self.layer_norm(M_updated)

        return M_updated


class MUFeedForward(nn.Module):
    """
    Position-wise feedforward network for MU matrices

    Args:
        r: Number of rows in MU matrix
        c: Number of columns in MU matrix
        d_ff: Feedforward hidden dimension
        dropout: Dropout probability
    """

    def __init__(self, r: int = 4, c: int = 4, d_ff: int = 64, dropout: float = 0.1):
        super().__init__()

        self.r = r
        self.c = c
        self.rc = r * c

        self.fc1 = nn.Linear(self.rc, d_ff)
        self.fc2 = nn.Linear(d_ff, self.rc)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([r, c], eps=1e-6)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M: [B, T, r, c]

        Returns:
            [B, T, r, c]
        """
        B, T, r, c = M.shape
        M_flat = M.view(B, T, self.rc)

        # Feedforward
        hidden = F.gelu(self.fc1(M_flat))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)

        # Residual connection
        M_out = M_flat + output
        M_out = M_out.view(B, T, r, c)

        # Layer norm
        M_out = self.layer_norm(M_out)

        return M_out


class MUTransformerBlock(nn.Module):
    """
    Complete MU Transformer block with attention and feedforward

    Args:
        r: Number of rows in MU matrix
        c: Number of columns in MU matrix
        d_model: Hidden dimension for attention
        n_heads: Number of attention heads
        d_ff: Feedforward hidden dimension
        dropout: Dropout probability
        sensitivity_mask: Optional sensitivity mask for gating
    """

    def __init__(
        self,
        r: int = 4,
        c: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 64,
        dropout: float = 0.1,
        sensitivity_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.attention = MUAttentionLayer(
            r=r,
            c=c,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            sensitivity_mask=sensitivity_mask
        )

        self.feedforward = MUFeedForward(
            r=r,
            c=c,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(
        self,
        M: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            M: [B, T, r, c]
            mask: Optional attention mask

        Returns:
            [B, T, r, c]
        """
        # Self-attention
        M = self.attention(M, mask=mask)

        # Feedforward
        M = self.feedforward(M)

        return M
