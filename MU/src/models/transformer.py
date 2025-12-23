"""MU-SOTA Transformer model"""

import torch
import torch.nn as nn
from typing import Optional
import logging

from .attention import BlockWiseSemanticAttention
from ..config import MUSOTAConfig

logger = logging.getLogger(__name__)


class MUSOTATransformer(nn.Module):
    """24-layer deep MU Transformer with semantic block structure"""

    def __init__(self, config: MUSOTAConfig):
        super().__init__()
        self.config = config

        # Token embeddings → 8×8 matrix
        self.token_to_mu = nn.Embedding(config.vocab_size, 64)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_seq_len, 64) * 0.02
        )

        # 24 layers of block-wise semantic attention (SOTA depth!)
        self.layers = nn.ModuleList([
            BlockWiseSemanticAttention(config)
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(64)

        # Output projection
        self.mu_to_logits = nn.Linear(64, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(f"Initialized MU-SOTA with {self.count_parameters():,} parameters")

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [B, T]
            mask: [B, T] (optional)

        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = input_ids.shape

        # Embed tokens to 8×8 matrices
        M = self.token_to_mu(input_ids)  # [B, T, 64]
        M = M + self.pos_encoding[:, :T, :]  # Add positional encoding
        M = M.reshape(B, T, 8, 8)  # Reshape to matrices

        # Process through 24 deep layers with structure-aware attention
        for layer in self.layers:
            M = layer(M, input_ids, mask)

        # Final norm and flatten
        M_flat = M.reshape(B, T, 64)
        M_flat = self.final_norm(M_flat)

        # Output logits
        logits = self.mu_to_logits(M_flat)

        return logits
