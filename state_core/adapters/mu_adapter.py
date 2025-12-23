"""
MU Adapter - Wraps MU semantic representation.

Imports MU exactly as implemented, outputs 8×8 semantic matrices.
Does NOT alter MU internals.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional

# Add MU repo to path
MU_PATH = Path(__file__).parent.parent.parent / "MU"
if str(MU_PATH) not in sys.path:
    sys.path.insert(0, str(MU_PATH))


class MUAdapter(nn.Module):
    """
    Adapter for MU semantic representation.
    
    Wraps MU's embedding and optionally transformer layers.
    Output: 8×8 semantic matrices [B, T, 8, 8] or flattened [B, T, 64]
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 64,  # 8x8 = 64
        max_seq_len: int = 512,
        flatten_output: bool = True,
        use_full_model: bool = False
    ):
        """
        Initialize MU adapter.
        
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension (64 for 8x8 matrix)
            max_seq_len: Maximum sequence length
            flatten_output: If True, output [B, T, 64], else [B, T, 8, 8]
            use_full_model: If True, use full MUSOTATransformer
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.flatten_output = flatten_output
        self.use_full_model = use_full_model
        
        if use_full_model:
            # Import and use full MU transformer
            try:
                from mu_sota import MUSOTATransformer, MUSOTAConfig
                config = MUSOTAConfig()
                config.vocab_size = vocab_size
                config.max_seq_len = max_seq_len
                self.mu_model = MUSOTATransformer(config)
                self._full_model = True
            except ImportError as e:
                print(f"Warning: Could not import MU model: {e}")
                print("Falling back to simple embedding")
                self._full_model = False
                self._init_simple_embedding(vocab_size, embed_dim, max_seq_len)
        else:
            self._full_model = False
            self._init_simple_embedding(vocab_size, embed_dim, max_seq_len)
    
    def _init_simple_embedding(self, vocab_size, embed_dim, max_seq_len):
        """
        Initialize simple semantic embedding (MU-style).
        
        NOTE: Semantic identity must remain invariant to sequence position.
        Positional information is tracked separately in State.position_indices
        and handled by TEMPORAL, NOT by MU.
        """
        # Token to 8x8 semantic matrix - pure semantic identity
        self.token_to_mu = nn.Embedding(vocab_size, embed_dim)
        
        # NO positional encoding - semantic identity is position-invariant
        # Position is temporal/structural, not semantic
        
        # Initialize
        nn.init.normal_(self.token_to_mu.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get semantic state from token IDs.
        
        IMPORTANT: This returns PURE semantic identity without positional
        encoding. Semantic meaning does NOT depend on sequence position.
        
        Args:
            token_ids: [B, T] token indices
            
        Returns:
            semantic_state: [B, T, 64] or [B, T, 8, 8]
        """
        if self._full_model:
            # Use full MU transformer (semantic only, no position)
            M = self.mu_model.token_to_mu(token_ids)
            # NOTE: Do NOT add positional encoding from MU model
        else:
            # Simple embedding - pure semantic identity
            M = self.token_to_mu(token_ids)  # [B, T, 64]
            # NO positional encoding added
        
        # Reshape to 8x8 if requested
        if not self.flatten_output:
            B, T = M.shape[:2]
            M = M.view(B, T, 8, 8)
        
        return M
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
