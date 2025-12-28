"""
Auxiliary task heads for block specialization.

This module implements lightweight task-specific heads that provide
supervision signals to guide semantic blocks toward interpretable roles.

Block assignments (from research):
- I (Identity): Entity typing / NER
- R2 (Relations): Dependency relations
- K (Knowledge): Part-of-speech tagging

The auxiliary losses are combined with the main LM objective to
encourage functional specialization without requiring explicit
multi-task training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def select_block(semantic_state: torch.Tensor, block_name: str) -> torch.Tensor:
    """
    Extract specific block from semantic state.
    
    Args:
        semantic_state: [B, T, 64] or [B*T, 64]
        block_name: One of ['I', 'R2', 'K', ...]
    
    Returns:
        block_features: [B*T, 4] - Features for specified block
    """
    # Block indices (from MU structure)
    block_map = {
        'I': 0,   # Identity
        'D': 1,   # Dynamics
        'R1': 2,  # Relation 1
        'R2': 3,  # Relation 2
        'K': 4,   # Knowledge
        'M': 5,   # Memory
        'T': 6,   # Temporal
        'P': 7,   # Process
        'S': 8,   # Spatial
        'C': 9,   # Causal
        'N': 10,  # Normative
        'X': 11,  # Context
        'E': 12,  # Emotional
        'F': 13,  # Function
        'A': 14,  # Action
        'Z': 15,  # Abstract
    }
    
    if block_name not in block_map:
        raise ValueError(f"Unknown block: {block_name}")
    
    block_idx = block_map[block_name]
    
    # Reshape to blocks if needed
    if semantic_state.dim() == 3:
        B, T, D = semantic_state.shape
        semantic_state = semantic_state.reshape(B * T, D)
    
    # Extract block (each block is 4D)
    block_start = block_idx * 4
    block_end = block_start + 4
    
    return semantic_state[:, block_start:block_end]


class AuxiliaryTaskLoss(nn.Module):
    """
    Multi-task auxiliary heads for block specialization.
    
    Provides supervision for:
    1. Part-of-speech tagging (K block)
    2. Named entity recognition (I block)
    3. Dependency relations (R2 block)
    
    Args:
        num_pos_tags: Number of POS tags (default: 17 for Universal POS)
        num_ner_tags: Number of NER tags (default: 9 for CoNLL-2003)
        num_dep_rels: Number of dependency relations (default: 40 for UD)
        block_dim: Dimension of each block (default: 4)
    """
    
    def __init__(
        self,
        num_pos_tags: int = 17,
        num_ner_tags: int = 9,
        num_dep_rels: int = 40,
        block_dim: int = 4
    ):
        super().__init__()
        
        # Task heads (simple linear classifiers)
        self.pos_head = nn.Linear(block_dim, num_pos_tags)   # K block → POS
        self.ner_head = nn.Linear(block_dim, num_ner_tags)   # I block → NER
        self.dep_head = nn.Linear(block_dim, num_dep_rels)   # R2 block → DEP
        
        # Task weights (for balancing)
        self.register_buffer('task_weights', torch.ones(3))
    
    def forward(
        self,
        semantic_state: torch.Tensor,
       labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute auxiliary task losses.
        
        Args:
            semantic_state: [B, T, 64] - Semantic block activations  
            labels: Dictionary with 'pos', 'ner', 'dep' labels (each [B, T])
                   If None, only computes predictions (for inference)
        
        Returns:
            loss: Combined auxiliary loss
            info: Dictionary with per-task losses and predictions
        """
        B, T, D = semantic_state.shape
        device = semantic_state.device
        
        # Extract block features
        I_block = select_block(semantic_state, 'I')    # [B*T, 4]
        R2_block = select_block(semantic_state, 'R2')  # [B*T, 4]
        K_block = select_block(semantic_state, 'K')    # [B*T, 4]
        
        # Compute predictions
        pos_logits = self.pos_head(K_block)   # [B*T, num_pos]
        ner_logits = self.ner_head(I_block)   # [B*T, num_ner]
        dep_logits = self.dep_head(R2_block)  # [B*T, num_dep]
        
        info = {
            'pos_logits': pos_logits.reshape(B, T, -1),
            'ner_logits': ner_logits.reshape(B, T, -1),
            'dep_logits': dep_logits.reshape(B, T, -1)
        }
        
        # If no labels, return zero loss (inference mode)
        if labels is None:
            return torch.tensor(0.0, device=device), info
        
        # Flatten labels
        pos_labels = labels['pos'].reshape(B * T)  # [B*T]
        ner_labels = labels['ner'].reshape(B * T)  # [B*T]
        dep_labels = labels['dep'].reshape(B * T)  # [B*T]
        
        # Compute losses (ignore padding tokens with label -100)
        pos_loss = F.cross_entropy(
            pos_logits,
            pos_labels,
            ignore_index=-100
        )
        ner_loss = F.cross_entropy(
            ner_logits,
            ner_labels,
            ignore_index=-100
        )
        dep_loss = F.cross_entropy(
            dep_logits,
            dep_labels,
            ignore_index=-100
        )
        
        # Combined loss (weighted)
        total_loss = (
            self.task_weights[0] * pos_loss +
            self.task_weights[1] * ner_loss +
            self.task_weights[2] * dep_loss
        ) / self.task_weights.sum()
        
        # Add individual losses to info
        info.update({
            'pos_loss': pos_loss.item(),
            'ner_loss': ner_loss.item(),
            'dep_loss': dep_loss.item(),
            'total_aux_loss': total_loss.item()
        })
        
        return total_loss, info
    
    def predict(self, semantic_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get predictions for auxiliary tasks (inference mode).
        
        Args:
            semantic_state: [B, T, 64]
        
        Returns:
            predictions: Dictionary with 'pos', 'ner', 'dep' predictions
        """
        _, info = self.forward(semantic_state, labels=None)
        
        return {
            'pos': info['pos_logits'].argmax(dim=-1),
            'ner': info['ner_logits'].argmax(dim=-1),
            'dep': info['dep_logits'].argmax(dim=-1)
        }


class BlockUsageBalancing(nn.Module):
    """
    Encourage uniform usage of all semantic blocks.
    
    Prevents scenario where some blocks dominate (e.g., block 0 used 90%,
    others 10%). Encourages all 16 blocks to contribute equally.
    
    Uses entropy maximization: uniform distribution has maximum entropy.
    """
    
    def __init__(self, num_blocks: int = 16, block_dim: int = 4):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.register_buffer('target_prob', torch.ones(num_blocks) / num_blocks)
    
    def forward(self, semantic_state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute block usage balancing loss.
        
        Args:
            semantic_state: [B, T, 64]
        
        Returns:
            loss: KL divergence from uniform distribution
            info: Usage statistics per block
        """
        B, T, D = semantic_state.shape
        
        # Reshape to blocks
        blocks = semantic_state.reshape(B, T, self.num_blocks, self.block_dim)
        
        # Compute block magnitudes (how much each block is "used")
        block_norms = blocks.norm(dim=3)  # [B, T, 16]
        
        # Softmax to get probability distribution
        block_probs = F.softmax(block_norms, dim=2)  # [B, T, 16]
        
        # Average across batch and time
        avg_usage = block_probs.mean(dim=[0, 1])  # [16]
        
        # KL divergence from uniform (encourages uniform usage)
        loss = F.kl_div(
            avg_usage.log(),
            self.target_prob,
            reduction='batchmean'
        )
        
        # Statistics
        info = {
            'block_usage': avg_usage.tolist(),
            'usage_entropy': -(avg_usage * avg_usage.log()).sum().item(),
            'max_usage': avg_usage.max().item(),
            'min_usage': avg_usage.min().item()
        }
        
        return loss, info
