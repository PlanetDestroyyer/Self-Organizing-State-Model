"""Block-wise semantic attention mechanism"""

import torch
import torch.nn as nn
from typing import Optional

from .semantic_blocks import SemanticBlockLayout
from .sensitivity import DynamicBlockSensitivity
from ..config import MUSOTAConfig


class BlockWiseSemanticAttention(nn.Module):
    """Structure-aware attention that processes semantic blocks separately"""

    def __init__(self, config: MUSOTAConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.num_blocks = config.num_semantic_blocks

        # Each semantic block gets its own attention module
        self.block_attentions = nn.ModuleDict()
        block_names = SemanticBlockLayout.get_all_block_names()

        for block_name in block_names:
            # Each 2×2 block = 4 values
            self.block_attentions[block_name] = nn.MultiheadAttention(
                embed_dim=4,
                num_heads=2,
                dropout=config.dropout,
                batch_first=True
            )

        # Cross-block attention for global refinement
        self.cross_block_attn = nn.MultiheadAttention(
            embed_dim=64,  # 8×8 matrix flattened
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 64)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)

        # Dynamic sensitivity computer
        self.sensitivity_computer = DynamicBlockSensitivity(
            config.num_semantic_blocks,
            config.vocab_size,
            64
        )

    def forward(self, M: torch.Tensor, token_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process 8×8 matrices with block-wise semantic attention

        Args:
            M: [B, T, 8, 8] - input matrices
            token_ids: [B, T] - for sensitivity computation
            mask: [B, T] - attention mask (optional)

        Returns:
            M_out: [B, T, 8, 8] - processed matrices
        """
        B, T = M.shape[0], M.shape[1]
        block_outputs = []

        # Process each semantic block independently (memory-efficient)
        for block_name in SemanticBlockLayout.get_all_block_names():
            r1, c1, r2, c2 = SemanticBlockLayout.get_block_indices(block_name)

            # Extract block
            block_data = M[:, :, r1:r2, c1:c2]  # [B, T, 2, 2]
            block_flat = block_data.reshape(B, T, 4)  # [B, T, 4]

            # Self-attention within block
            block_out, _ = self.block_attentions[block_name](
                block_flat, block_flat, block_flat,
                key_padding_mask=mask if mask is not None else None
            )

            block_outputs.append(block_out)

            # Clear intermediate tensors to save memory
            del block_data, block_flat

        # Combine all blocks
        all_blocks = torch.cat(block_outputs, dim=-1)  # [B, T, 64]
        del block_outputs  # Free memory

        # Cross-block attention (blocks interact)
        cross_out, attn_weights = self.cross_block_attn(
            all_blocks, all_blocks, all_blocks,
            key_padding_mask=mask if mask is not None else None
        )

        # Residual + norm
        all_blocks = self.norm1(all_blocks + cross_out)

        # Feed-forward
        ffn_out = self.ffn(all_blocks)
        all_blocks = self.norm2(all_blocks + ffn_out)

        # Compute dynamic sensitivity
        sensitivity = self.sensitivity_computer(token_ids, attn_weights)  # [B, T, 16]

        # Apply sensitivity-based gating (block-wise)
        all_blocks_reshaped = all_blocks.reshape(B, T, 16, 4)  # [B, T, 16 blocks, 4 values]
        sensitivity_expanded = sensitivity.unsqueeze(-1)  # [B, T, 16, 1]

        # Modulate each block by its sensitivity
        M_flat_original = M.reshape(B, T, 16, 4)
        delta = all_blocks_reshaped - M_flat_original
        M_flat_new = M_flat_original + delta * sensitivity_expanded

        # Reshape back to 8×8
        M_out = M_flat_new.reshape(B, T, 8, 8)

        return M_out
