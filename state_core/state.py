"""
Central State Object for the Self-Organizing State Model.

This is the core identity of the model. All data flows through State.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class State:
    """
    Central state object that flows through the system.
    
    Attributes:
        semantic_state: MU matrices [B, T, 64] or [B, T, 8, 8]
        temporal_state: TEMPORAL embeddings [B, T, time_dim]
        routing_state: Graph structure (adjacency, mask)
        responsibility: K-1 attribution info
        metadata: Additional logging data
    """
    semantic_state: Optional[torch.Tensor] = None
    temporal_state: Optional[torch.Tensor] = None
    routing_state: Optional[Dict[str, Any]] = None
    responsibility: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def combined_embedding(self) -> torch.Tensor:
        """Get combined semantic + temporal embedding."""
        if self.temporal_state is None:
            return self.semantic_state
        
        # Flatten semantic if needed [B, T, 8, 8] -> [B, T, 64]
        semantic = self.semantic_state
        if semantic.dim() == 4:
            B, T = semantic.shape[:2]
            semantic = semantic.view(B, T, -1)
        
        # Concatenate
        return torch.cat([semantic, self.temporal_state], dim=-1)
    
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
        if self.routing_state is not None:
            summary['graph_edges'] = self.routing_state.get('num_edges', 0)
        if self.responsibility is not None:
            summary['nodes_updated'] = self.responsibility.get('nodes_updated', 0)
            summary['update_pct'] = self.responsibility.get('update_pct', 0)
        
        return summary
