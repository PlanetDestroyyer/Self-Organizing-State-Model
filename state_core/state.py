"""
Central State Object for the Self-Organizing State Model.

This is the core identity of the model. All data flows through State.

DESIGN PRINCIPLES:
- Semantic and temporal states are NEVER concatenated
- Position is tracked explicitly via position_indices
- State carries all context needed for computation
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class State:
    """
    Central state object that flows through the system.
    
    IMPORTANT: Semantic and temporal states remain SEPARATE.
    They are only combined internally by the StateUpdateOperator
    for computation purposes, never in the State itself.
    
    Attributes:
        semantic_state: MU matrices [B, T, 64] - pure semantic identity
        temporal_state: TEMPORAL embeddings [B, T, time_dim]
        position_indices: Explicit sequence positions [B, T]
        routing_state: Graph structure (adjacency, mask)
        responsibility: K-1 attribution info
        metadata: Additional logging data
    """
    semantic_state: Optional[torch.Tensor] = None
    temporal_state: Optional[torch.Tensor] = None
    position_indices: Optional[torch.Tensor] = None  # Explicit position tracking
    routing_state: Optional[Dict[str, Any]] = None
    responsibility: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # NOTE: combined_embedding() method REMOVED
    # Semantic and temporal states must remain interpretably separate
    # The StateUpdateOperator handles internal projection for computation
    
    def get_mu_identity_block(self) -> Optional[torch.Tensor]:
        """
        Get MU Identity block (first 4 elements of semantic state).
        Used by GraphBuilder for semantic similarity edges.
        """
        if self.semantic_state is None:
            return None
        # Identity block is first 4 dimensions (4 values in 8×8 matrix)
        return self.semantic_state[..., :4]
    
    def get_attention_mask(self) -> Optional[torch.Tensor]:
        """Get attention mask from routing state."""
        if self.routing_state is None:
            return None
        return self.routing_state.get('attention_mask')
    
    def log_summary(self) -> Dict[str, Any]:
        """Get summary for logging."""
        summary = {}
        
        if self.semantic_state is not None:
            summary['semantic_shape'] = list(self.semantic_state.shape)
        if self.temporal_state is not None:
            summary['temporal_shape'] = list(self.temporal_state.shape)
        if self.position_indices is not None:
            summary['positions'] = True
        if self.routing_state is not None:
            summary['graph_edges'] = self.routing_state.get('num_edges', 0)
        if self.responsibility is not None:
            summary['nodes_updated'] = self.responsibility.get('nodes_updated', 0)
            summary['update_pct'] = self.responsibility.get('update_pct', 0)
        
        return summary

