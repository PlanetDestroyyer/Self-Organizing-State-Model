"""
Graph Mask Converter - Converts graph adjacency to attention mask.

Applies mask BEFORE softmax in attention layers.
Always allows self-attention.
Never modifies embeddings directly.
"""

import torch
from typing import Dict, Any, List, Tuple


class GraphMaskConverter:
    """
    Converts graph adjacency list to attention mask tensor.
    
    The mask is applied to attention scores before softmax:
        masked_scores = scores + mask  (where mask has -inf for blocked)
        OR
        masked_scores = scores.masked_fill(~mask, -inf)
    """
    
    def __init__(self, mask_value: float = -1e9):
        """
        Initialize converter.
        
        Args:
            mask_value: Value for blocked attention (use -inf or -1e9)
        """
        self.mask_value = mask_value
    
    def convert(
        self,
        graph: Dict[str, Any],
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Convert graph to attention mask.
        
        Args:
            graph: Dict from GraphBuilder.build_graph()
            device: Target device
            
        Returns:
            attention_mask: [T, T] boolean mask (True = allowed)
        """
        seq_len = graph['seq_len']
        adjacency = graph['adjacency']
        
        # Create mask (start with all blocked)
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        # Enable edges from adjacency list
        for i, j in adjacency:
            if 0 <= i < seq_len and 0 <= j < seq_len:
                mask[i, j] = True
        
        if device is not None:
            mask = mask.to(device)
        
        return mask
    
    def convert_to_additive(
        self,
        graph: Dict[str, Any],
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Convert to additive mask (add to attention scores).
        
        Args:
            graph: Dict from GraphBuilder.build_graph()
            device: Target device
            
        Returns:
            additive_mask: [T, T] float mask (0 = allowed, -inf = blocked)
        """
        bool_mask = self.convert(graph, device)
        
        # Convert: True -> 0, False -> -inf
        additive_mask = torch.zeros_like(bool_mask, dtype=torch.float32)
        additive_mask = additive_mask.masked_fill(~bool_mask, self.mask_value)
        
        return additive_mask
    
    def convert_batched(
        self,
        graphs: List[Dict[str, Any]],
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Convert multiple graphs to batched attention mask.
        
        Args:
            graphs: List of graph dicts
            device: Target device
            
        Returns:
            attention_mask: [B, T, T] boolean mask
        """
        batch_size = len(graphs)
        seq_len = graphs[0]['seq_len']
        
        masks = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        
        for b, graph in enumerate(graphs):
            for i, j in graph['adjacency']:
                if 0 <= i < seq_len and 0 <= j < seq_len:
                    masks[b, i, j] = True
        
        if device is not None:
            masks = masks.to(device)
        
        return masks
    
    def apply_to_scores(
        self,
        attention_scores: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mask to attention scores.
        
        Args:
            attention_scores: [B, H, T, T] or [B, T, T]
            mask: [T, T] or [B, T, T] boolean mask
            
        Returns:
            masked_scores: Same shape as input
        """
        # Expand mask dimensions if needed
        if attention_scores.dim() == 4 and mask.dim() == 2:
            # [T, T] -> [1, 1, T, T]
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif attention_scores.dim() == 4 and mask.dim() == 3:
            # [B, T, T] -> [B, 1, T, T]
            mask = mask.unsqueeze(1)
        
        # Apply mask
        return attention_scores.masked_fill(~mask, self.mask_value)
