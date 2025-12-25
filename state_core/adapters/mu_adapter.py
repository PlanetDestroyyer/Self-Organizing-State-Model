"""
MU Adapter - Full Implementation with 16 Semantic Blocks

Integrates MU's block-wise semantic attention for rich token representation.
Each token is an 8×8 matrix with 16 semantic blocks.

Blocks:
- I (Identity): Core token meaning
- S (Structure): Grammatical properties
- C1/C2 (Context): Local/global context
- R1/R2 (Relations): Syntactic/semantic dependencies
- T (Transform): Compositional changes
- K (Knowledge): World knowledge
- G (Global): Document coherence
- M (Modality): Certainty/tense
- D (Discourse): Rhetorical structure
- F (Frame): Semantic frame roles
- P (Position): Positional encoding (learned)
- E (Entity): Named entity properties
- A (Affect): Sentiment/emotion
- X (Extension): Flexible/learned purpose
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

# Add MU repo to path
MU_PATH = Path(__file__).parent.parent.parent / "MU"
if str(MU_PATH) not in sys.path:
    sys.path.insert(0, str(MU_PATH))


# ============================================================================
# SEMANTIC BLOCK DEFINITIONS
# ============================================================================

class SemanticBlockLayout:
    """Defines the 16 semantic blocks in 8×8 matrix"""
    
    BLOCKS = {
        # Name: (row_start, col_start, row_end, col_end, description)
        'I': (0, 0, 2, 2, 'Identity - core token meaning'),
        'S': (0, 2, 2, 4, 'Structure - grammatical properties'),
        'C1': (0, 4, 2, 6, 'Context-Local - immediate context'),
        'C2': (0, 6, 2, 8, 'Context-Global - document context'),
        'R1': (2, 0, 4, 2, 'Relations-Syntactic'),
        'R2': (2, 2, 4, 4, 'Relations-Semantic'),
        'T': (2, 4, 4, 6, 'Transformation - compositional'),
        'K': (2, 6, 4, 8, 'Knowledge - world knowledge'),
        'G': (4, 0, 6, 2, 'Global - document coherence'),
        'M': (4, 2, 6, 4, 'Modality - certainty/tense'),
        'D': (4, 4, 6, 6, 'Discourse - rhetorical'),
        'F': (4, 6, 6, 8, 'Frame - semantic roles'),
        'P': (6, 0, 8, 2, 'Position - positional'),
        'E': (6, 2, 8, 4, 'Entity - named entities'),
        'A': (6, 4, 8, 6, 'Affect - sentiment'),
        'X': (6, 6, 8, 8, 'Extension - learned'),
    }
    
    @classmethod
    def get_block_indices(cls, block_name: str) -> Tuple[int, int, int, int]:
        return cls.BLOCKS[block_name][:4]
    
    @classmethod
    def get_all_block_names(cls) -> List[str]:
        return list(cls.BLOCKS.keys())
    
    @classmethod
    def get_flat_indices(cls, block_name: str) -> List[int]:
        """Get indices in flattened 64-dim vector for a block."""
        r1, c1, r2, c2 = cls.get_block_indices(block_name)
        indices = []
        for r in range(r1, r2):
            for c in range(c1, c2):
                indices.append(r * 8 + c)
        return indices


# ============================================================================
# DYNAMIC SENSITIVITY (Learned gating per block)
# ============================================================================

class DynamicSensitivity(nn.Module):
    """Compute sensitivity for each semantic block - ALL LEARNED"""
    
    def __init__(self, num_blocks: int, vocab_size: int):
        super().__init__()
        self.num_blocks = num_blocks
        
        # LEARNED: Base sensitivity for each block
        self.block_sensitivity_base = nn.Parameter(
            torch.randn(num_blocks) * 0.1 + 0.5
        )
        
        # LEARNED: Token affinity to each block
        self.token_block_affinity = nn.Embedding(vocab_size, num_blocks)
        nn.init.normal_(self.token_block_affinity.weight, mean=0.0, std=0.1)
        
        # LEARNED: Sensitivity modulation network
        self.sensitivity_net = nn.Sequential(
            nn.Linear(num_blocks, num_blocks * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(num_blocks * 2, num_blocks),
            nn.Sigmoid()
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute block-wise sensitivity.
        
        Args:
            token_ids: [B, T]
            
        Returns:
            sensitivity: [B, T, num_blocks]
        """
        # Token-specific block affinity
        affinity = self.token_block_affinity(token_ids)  # [B, T, 16]
        
        # Compute sensitivity through learned network
        sensitivity = self.sensitivity_net(affinity)  # [B, T, 16]
        
        # Modulate by base sensitivity
        base = self.block_sensitivity_base.view(1, 1, -1)
        sensitivity = sensitivity * base
        
        return sensitivity


# ============================================================================
# BLOCK-WISE SEMANTIC ATTENTION
# ============================================================================

class BlockWiseAttention(nn.Module):
    """
    Structure-aware attention that processes semantic blocks separately.
    Each of the 16 blocks gets its own attention module.
    """
    
    def __init__(self, vocab_size: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_blocks = 16
        
        # Each semantic block gets its own attention module
        self.block_attentions = nn.ModuleDict()
        for block_name in SemanticBlockLayout.get_all_block_names():
            # Each 2×2 block = 4 values
            self.block_attentions[block_name] = nn.MultiheadAttention(
                embed_dim=4,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Cross-block attention for global refinement
        self.cross_block_attn = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        
        # Dynamic sensitivity
        self.sensitivity = DynamicSensitivity(16, vocab_size)
    
    def forward(
        self,
        M: torch.Tensor,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process 8×8 matrices with block-wise semantic attention.
        
        Args:
            M: [B, T, 8, 8] input matrices
            token_ids: [B, T] for sensitivity computation
            mask: [B, T] attention mask
            
        Returns:
            M_out: [B, T, 8, 8] processed matrices
        """
        B, T = M.shape[0], M.shape[1]
        block_outputs = {}
        
        # Process each semantic block independently
        for block_name in SemanticBlockLayout.get_all_block_names():
            r1, c1, r2, c2 = SemanticBlockLayout.get_block_indices(block_name)
            
            # Extract block
            block_data = M[:, :, r1:r2, c1:c2]  # [B, T, 2, 2]
            block_flat = block_data.reshape(B, T, 4)  # [B, T, 4]
            
            # Self-attention within block
            block_out, _ = self.block_attentions[block_name](
                block_flat, block_flat, block_flat,
                key_padding_mask=mask
            )
            
            block_outputs[block_name] = block_out
        
        # Combine all blocks [B, T, 64]
        all_blocks = torch.cat(list(block_outputs.values()), dim=-1)
        
        # Pre-LayerNorm: Normalize BEFORE attention
        # This improves gradient flow and training stability
        all_blocks_normed = self.norm1(all_blocks)
        cross_out, _ = self.cross_block_attn(
            all_blocks_normed, all_blocks_normed, all_blocks_normed,
            key_padding_mask=mask
        )
        
        # Residual connection
        all_blocks = all_blocks + cross_out
        
        # Pre-LayerNorm for FFN
        all_blocks_normed = self.norm2(all_blocks)
        ffn_out = self.ffn(all_blocks_normed)
        all_blocks = all_blocks + ffn_out
        
        # Compute dynamic sensitivity
        sensitivity = self.sensitivity(token_ids)  # [B, T, 16]
        
        # Apply sensitivity-based gating
        all_blocks_reshaped = all_blocks.reshape(B, T, 16, 4)
        sensitivity_expanded = sensitivity.unsqueeze(-1)
        
        # Modulate each block by its sensitivity
        M_flat_original = M.reshape(B, T, 16, 4)
        delta = all_blocks_reshaped - M_flat_original
        M_flat_new = M_flat_original + delta * sensitivity_expanded
        
        # Reshape back to 8×8
        M_out = M_flat_new.reshape(B, T, 8, 8)
        
        return M_out


# ============================================================================
# MU ADAPTER (Main Class)
# ============================================================================

class MUAdapter(nn.Module):
    """
    Full MU Adapter with 16 Semantic Blocks.
    
    Features:
    - Token embedding → 8×8 semantic matrix
    - Block-wise attention (16 separate attention modules)
    - Dynamic sensitivity gating
    - Cross-block refinement
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 64,
        max_seq_len: int = 512,
        flatten_output: bool = True,
        use_full_model: bool = True,  # Now True by default!
        n_layers: int = 2,  # Number of block attention layers
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.flatten_output = flatten_output
        self.use_full_model = use_full_model
        
        # Token embedding → 64-dim (8×8 flattened)
        self.token_to_mu = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.token_to_mu.weight, mean=0.0, std=0.02)
        
        if use_full_model:
            # Full block-wise attention layers
            self.block_layers = nn.ModuleList([
                BlockWiseAttention(vocab_size, n_heads=2, dropout=dropout)
                for _ in range(n_layers)
            ])
            self._full_model = True
            print(f"✓ MU Adapter initialized with {n_layers} block-attention layers (16 semantic blocks)")
        else:
            self._full_model = False
            print("  MU Adapter using simple embedding (no block attention)")
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get semantic state from token IDs.

        Args:
            token_ids: [B, T]

        Returns:
            semantic_state: [B, T, embed_dim] or [B, T, 8, 8]
        """
        B, T = token_ids.shape

        # Initial embedding
        M = self.token_to_mu(token_ids)  # [B, T, embed_dim]

        if self._full_model:
            # Full model requires 64D (8×8 structure)
            if self.embed_dim != 64:
                raise ValueError(f"Full MU model requires embed_dim=64 for 8×8 structure, got {self.embed_dim}")

            # Reshape to 8×8 for block processing
            M = M.view(B, T, 8, 8)

            # Process through block attention layers
            for layer in self.block_layers:
                M = layer(M, token_ids)

            # Flatten if needed
            if self.flatten_output:
                M = M.view(B, T, 64)
        else:
            # Simple embedding path - use configured dimension
            # No reshaping needed, just return as [B, T, embed_dim]
            pass

        return M
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def get_block_layout(self) -> Dict:
        """Return semantic block layout for interpretability."""
        return SemanticBlockLayout.BLOCKS
