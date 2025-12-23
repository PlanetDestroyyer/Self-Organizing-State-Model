"""Dynamic block sensitivity computation - fully learned, no hardcoding"""

import torch
import torch.nn as nn
from typing import Optional


class DynamicBlockSensitivity(nn.Module):
    """Compute sensitivity for each semantic block - ALL LEARNED"""

    def __init__(self, num_blocks: int, vocab_size: int, d_model: int):
        super().__init__()
        self.num_blocks = num_blocks

        # LEARNED: Base sensitivity for each block (not hardcoded!)
        self.block_sensitivity_base = nn.Parameter(
            torch.randn(num_blocks) * 0.1 + 0.5  # Init ~0.5, then learn
        )

        # LEARNED: Token affinity to each block
        self.token_block_affinity = nn.Parameter(
            torch.randn(vocab_size, num_blocks) * 0.1
        )

        # LEARNED: Block interaction matrix
        self.block_interaction = nn.Parameter(
            torch.eye(num_blocks) * 0.5 + torch.randn(num_blocks, num_blocks) * 0.1
        )

        # LEARNED: Sensitivity modulation network
        self.sensitivity_net = nn.Sequential(
            nn.Linear(num_blocks + 1, num_blocks * 2),  # +1 for attention entropy
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(num_blocks * 2, num_blocks),
            nn.Sigmoid()  # 0-1 range
        )

    def forward(self, token_ids: torch.Tensor, attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute block-wise sensitivity - all from learned parameters!

        Args:
            token_ids: [B, T]
            attention_weights: [B, T, T] - from MultiheadAttention (optional)

        Returns:
            sensitivity: [B, T, num_blocks] - fully computed, no hardcoding!
        """
        B, T = token_ids.shape

        # Token-specific block affinity (LEARNED)
        affinity = self.token_block_affinity[token_ids]  # [B, T, num_blocks]

        # Attention-based modulation (COMPUTED from context)
        if attention_weights is not None:
            # attention_weights: [B, T, T] -> compute entropy per token
            attn_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-9),
                dim=-1
            )  # [B, T] - entropy for each token position
        else:
            attn_entropy = torch.zeros(B, T, device=token_ids.device)

        # Combine features
        features = torch.cat([affinity, attn_entropy.unsqueeze(-1)], dim=-1)  # [B, T, num_blocks+1]

        # Compute sensitivity through learned network
        sensitivity = self.sensitivity_net(features)  # [B, T, num_blocks]

        # Modulate by base sensitivity (LEARNED parameter)
        base = self.block_sensitivity_base.view(1, 1, -1)
        sensitivity = sensitivity * base

        return sensitivity  # All values learned or computed - NO HARDCODING!
