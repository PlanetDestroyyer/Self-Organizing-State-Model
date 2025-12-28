"""
Contrastive loss for same token in different contexts.

This module implements context-aware contrastive learning to push
semantic block representations apart when the same token appears
in different semantic contexts.

Example:
    "The bank by the river" vs "The bank approved my loan"
    → Same token "bank", different contexts
    → Should have different block activations

Based on SimCLR/MoCo but adapted for structured semantic blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ContextContrastiveLoss(nn.Module):
    """
    Contrastive loss for context-dependent semantic differentiation.
    
    Methodology:
    1. Find tokens that appear multiple times in batch (same token ID)
    2. Compute block-level embeddings for each occurrence
    3. Contrast based on context similarity:
       - Similar contexts → pull together
       - Different contexts → push apart
    
    Args:
        temperature: Temperature parameter for contrastive loss (default: 0.07)
        context_window: Window size for context comparison (default: 3)
        min_occurrences: Minimum token occurrences to apply loss (default: 2)
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        context_window: int = 3,
        min_occurrences: int = 2
    ):
        super().__init__()
        self.temperature = temperature
        self.context_window = context_window
        self.min_occurrences = min_occurrences
    
    def forward(
        self,
        semantic_state: torch.Tensor,
        token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute contrastive loss for repeated tokens.
        
        Args:
            semantic_state: [B, T, 64] - Semantic block activations
            token_ids: [B, T] - Token IDs
        
        Returns:
            loss: Scalar contrastive loss
            info: Dictionary with statistics
        """
        B, T, D = semantic_state.shape
        device = semantic_state.device
        
        # FAST PATH: Just compute variance across batch
        # If same token appears multiple times, we want high variance
        # (different contexts → different representations)
        
        # Flatten batch dimension
        semantic_flat = semantic_state.reshape(B * T, D)  # [B*T, 64]
        tokens_flat = token_ids.reshape(B * T)  # [B*T]
        
        # Find repeated tokens (fast)
        unique_tokens, inverse_indices, counts = torch.unique(
            tokens_flat, 
            return_inverse=True, 
            return_counts=True
        )
        
        # Only process tokens that appear 2+ times
        repeated_mask = counts >= self.min_occurrences
        
        if not repeated_mask.any():
            return torch.tensor(0.0, device=device), {
                'num_pairs': 0,
                'num_repeated_tokens': 0
            }
        
        # For repeated tokens, compute variance of their representations
        # High variance = good (different contexts → different states)
        # Low variance = bad (same representation despite different contexts)
        
        total_variance = 0.0
        num_repeated = 0
        
        for token_idx in repeated_mask.nonzero(as_tuple=True)[0]:
            # Get all positions of this token
            mask = (inverse_indices == token_idx)
            token_reps = semantic_flat[mask]  # [N, 64]
            
            if token_reps.shape[0] < 2:
                continue
            
            # Compute variance across occurrences
            # High variance = different contexts have different reps (GOOD)
            variance = token_reps.var(dim=0).mean()  # Average variance across dimensions
            
            # Loss: minimize negative variance = maximize variance
            total_variance += variance
            num_repeated += 1
        
        if num_repeated == 0:
            return torch.tensor(0.0, device=device), {
                'num_pairs': 0,
                'num_repeated_tokens': 0
            }
        
        # Average variance
        avg_variance = total_variance / num_repeated
        
        # Loss: we want HIGH variance, so minimize -variance
        # Or equivalently, add variance as a bonus (negative loss)
        # But contrastive losses are usually positive, so:
        # Use target_variance - actual_variance
        target_variance = 1.0  # Target: high variance
        loss = torch.clamp(target_variance - avg_variance, min=0.0)
        
        # Statistics
        info = {
            'num_repeated_tokens': num_repeated,
            'avg_variance': avg_variance.item()
        }
        
        return loss, info


class BlockContrastiveLoss(nn.Module):
    """
    Alternative: Block-level contrastive loss.
    
    Instead of contrasting entire 64D vectors, contrast at block level.
    This allows finer-grained differentiation.
    
    For same token in different contexts:
    - Some blocks might stay similar (e.g., identity)
    - Other blocks should differ (e.g., semantic role)
    """
    
    def __init__(self, temperature: float = 0.07, num_blocks: int = 16):
        super().__init__()
        self.temperature = temperature
        self.num_blocks = num_blocks
        self.block_dim = 4
    
    def forward(
        self,
        semantic_state: torch.Tensor,
        token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute block-level contrastive loss.
        
        Args:
            semantic_state: [B, T, 64]
            token_ids: [B, T]
        
        Returns:
            loss: Contrastive loss at block level
            info: Statistics
        """
        B, T, D = semantic_state.shape
        device = semantic_state.device
        
        # Reshape to blocks
        semantic_blocks = semantic_state.reshape(B, T, self.num_blocks, self.block_dim)
        semantic_flat = semantic_blocks.reshape(B * T, self.num_blocks, self.block_dim)
        tokens_flat = token_ids.reshape(B * T)
        
        # Find repeated tokens
        unique_tokens, counts = torch.unique(tokens_flat, return_counts=True)
        repeated_mask = counts >= 2
        repeated_tokens = unique_tokens[repeated_mask]
        
        if len(repeated_tokens) == 0:
            return torch.tensor(0.0, device=device), {'num_pairs': 0}
        
        total_loss = 0.0
        num_pairs = 0
        
        for token_id in repeated_tokens:
            positions = (tokens_flat == token_id).nonzero(as_tuple=True)[0]
            
            if len(positions) < 2:
                continue
            
            # Get block embeddings for all occurrences
            token_blocks = semantic_flat[positions]  # [N, 16, 4]
            N = len(positions)
            
            # Normalize per block
            token_blocks = F.normalize(token_blocks, p=2, dim=2)  # [N, 16, 4]
            
            # Compute block-wise similarity for all pairs
            for i in range(N):
                for j in range(i + 1, N):
                    # Similarity per block
                    block_sims = (token_blocks[i] * token_blocks[j]).sum(dim=1)  # [16]
                    
                    # Loss: encourage low similarity (push apart)
                    pair_loss = block_sims.mean()
                    total_loss += pair_loss
                    num_pairs += 1
        
        if num_pairs == 0:
            return torch.tensor(0.0, device=device), {'num_pairs': 0}
        
        loss = total_loss / num_pairs
        
        return loss, {'num_pairs': num_pairs}
