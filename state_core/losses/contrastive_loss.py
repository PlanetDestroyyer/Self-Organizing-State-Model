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
        
        # Flatten batch dimension for easier processing
        semantic_flat = semantic_state.reshape(B * T, D)  # [B*T, 64]
        tokens_flat = token_ids.reshape(B * T)  # [B*T]
        
        # Find repeated tokens
        unique_tokens, counts = torch.unique(tokens_flat, return_counts=True)
        repeated_mask = counts >= self.min_occurrences
        repeated_tokens = unique_tokens[repeated_mask]
        
        if len(repeated_tokens) == 0:
            # No repeated tokens, return zero loss
            return torch.tensor(0.0, device=device), {
                'num_pairs': 0,
                'avg_similarity': 0.0
            }
        
        total_loss = 0.0
        num_pairs = 0
        total_sim = 0.0
        
        # For each repeated token
        for token_id in repeated_tokens:
            # Find all positions where this token appears
            positions = (tokens_flat == token_id).nonzero(as_tuple=True)[0]
            
            if len(positions) < 2:
                continue
            
            # Get embeddings for all occurrences
            token_embeddings = semantic_flat[positions]  # [N, 64]
            
            # Normalize embeddings
            token_embeddings = F.normalize(token_embeddings, p=2, dim=1)
            
            # Compute pairwise similarity
            similarity_matrix = torch.mm(
                token_embeddings,
                token_embeddings.t()
            )  # [N, N]
            
            # Create labels: diagonal is positive (same instance)
            # Off-diagonal is negative (different context)
            N = len(positions)
            
            # For simplicity: treat each occurrence as anchor
            # Contrast with all other occurrences of same token
            # (Different contexts should push apart)
            for i in range(N):
                # Anchor
                anchor_sim = similarity_matrix[i]  # [N]
                
                # Positive: same instance (just itself, set to high value)
                anchor_sim[i] = 1.0 / self.temperature
                
                # Negatives: all other occurrences (different contexts)
                negatives = anchor_sim.clone()
                negatives[i] = -float('inf')  # Mask self
                
                # InfoNCE-style loss (maximize dissimilarity with other contexts)
                # We want different contexts to be dissimilar
                # So we minimize similarity
                context_loss = -torch.logsumexp(
                    -negatives / self.temperature,
                    dim=0
                )
                
                total_loss += context_loss
                num_pairs += (N - 1)
                total_sim += anchor_sim[anchor_sim != 1.0/self.temperature].mean().item()
        
        if num_pairs == 0:
            return torch.tensor(0.0, device=device), {
                'num_pairs': 0,
                'avg_similarity': 0.0
            }
        
        # Average loss
        loss = total_loss / num_pairs
        
        # Statistics
        info = {
            'num_pairs': num_pairs,
            'num_repeated_tokens': len(repeated_tokens),
            'avg_similarity': total_sim / num_pairs if num_pairs > 0 else 0.0
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
