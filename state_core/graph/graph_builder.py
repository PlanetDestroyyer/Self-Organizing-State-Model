"""
Graph Builder - Constructs per-sequence graphs for attention routing.

Builds graphs with:
- Sequential edges (always): i ↔ i+1
- Semantic similarity edges (optional): based on MU Identity block
- Small-world shortcuts (optional): random long-range connections
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import random


class GraphBuilder:
    """
    Builds per-sequence token graphs for attention routing.
    
    The graph determines which tokens can attend to which other tokens.
    """
    
    def __init__(
        self,
        enable_sequential: bool = True,
        enable_semantic: bool = False,
        enable_shortcuts: bool = False,
        semantic_threshold: float = 0.5,
        semantic_k: int = 5,  # NEW: Top-K edges per token
        semantic_method: str = 'topk',  # NEW: 'topk' or 'threshold'
        shortcut_prob: float = 0.05,
        bidirectional: bool = True
    ):
        """
        Initialize graph builder.
        
        Args:
            enable_sequential: Include i↔i+1 edges (always recommended)
            enable_semantic: Include semantic similarity edges
            enable_shortcuts: Include random small-world shortcuts
            semantic_threshold: Minimum similarity for semantic edge (threshold method)
            semantic_k: Number of top similar tokens to connect (topk method)
            semantic_method: 'topk' or 'threshold' - method for semantic edges
            shortcut_prob: Probability of random shortcut
            bidirectional: Whether edges are bidirectional
        """
        self.enable_sequential = enable_sequential
        self.enable_semantic = enable_semantic
        self.enable_shortcuts = enable_shortcuts
        self.semantic_threshold = semantic_threshold
        self.semantic_k = semantic_k
        self.semantic_method = semantic_method
        self.shortcut_prob = shortcut_prob
        self.bidirectional = bidirectional
    
    def build_graph(
        self,
        seq_len: int,
        semantic_state: Optional[torch.Tensor] = None,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Build graph for a sequence.
        
        Args:
            seq_len: Sequence length
            semantic_state: [B, T, D] semantic embeddings for similarity
            batch_size: Batch size
            
        Returns:
            Dict with:
                - adjacency: List of (i, j) edges
                - num_edges: Number of edges
                - edge_types: Dict mapping edge type to count
        """
        adjacency: List[Tuple[int, int]] = []
        edge_types = {'sequential': 0, 'semantic': 0, 'shortcut': 0}
        
        # Sequential edges: i ↔ i+1
        if self.enable_sequential:
            for i in range(seq_len - 1):
                adjacency.append((i, i + 1))
                edge_types['sequential'] += 1
                if self.bidirectional:
                    adjacency.append((i + 1, i))
                    edge_types['sequential'] += 1
        
        # Self-attention: i → i (always allowed)
        for i in range(seq_len):
            adjacency.append((i, i))
        
        # Semantic similarity edges
        if self.enable_semantic and semantic_state is not None:
            if self.semantic_method == 'topk':
                semantic_edges = self._build_semantic_edges_topk(semantic_state, seq_len)
            else:  # threshold method
                semantic_edges = self._build_semantic_edges_threshold(semantic_state, seq_len)
            adjacency.extend(semantic_edges)
            edge_types['semantic'] = len(semantic_edges)
        
        # Small-world shortcuts
        if self.enable_shortcuts:
            shortcut_edges = self._build_shortcuts(seq_len)
            adjacency.extend(shortcut_edges)
            edge_types['shortcut'] = len(shortcut_edges)
        
        return {
            'adjacency': adjacency,
            'num_edges': len(adjacency),
            'edge_types': edge_types,
            'seq_len': seq_len,
            'batch_size': batch_size
        }
    
    
    def _build_semantic_edges_topk(
        self,
        semantic_state: torch.Tensor,
        seq_len: int
    ) -> List[Tuple[int, int]]:
        """
        Build top-K semantic edges per token.
        
        Each token connects to its K most similar tokens (excluding self and adjacent).
        """
        edges = []
        
        # Use first batch item for edge computation
        state = semantic_state[0] if semantic_state.dim() == 3 else semantic_state
        
        # Normalize for cosine similarity
        state_norm = F.normalize(state, dim=-1)
        
        # Compute pairwise similarity [T, T]
        similarity = torch.mm(state_norm, state_norm.t())
        
        for i in range(seq_len):
            # Mask out self and adjacent tokens
            sim_i = similarity[i].clone()
            sim_i[i] = -1.0  # Mask self
            if i > 0:
                sim_i[i-1] = -1.0  # Mask previous (sequential edge exists)
            if i < seq_len - 1:
                sim_i[i+1] = -1.0  # Mask next (sequential edge exists)
            
            # Get top-K most similar
            k = min(self.semantic_k, seq_len - 3)  # Can't exceed available tokens
            if k > 0:
                top_k_sim, top_k_idx = sim_i.topk(k)
                
                for j_idx, sim_val in zip(top_k_idx, top_k_sim):
                    j = j_idx.item()
                    
                    # Optional: still apply minimum threshold
                    if self.semantic_threshold > 0 and sim_val < self.semantic_threshold:
                        continue
                    
                    # Add edge only once (i < j to avoid duplicates)
                    if j > i:
                        edges.append((i, j))
                        if self.bidirectional:
                            edges.append((j, i))
        
        return edges
    
    def _build_semantic_edges_threshold(
        self,
        semantic_state: torch.Tensor,
        seq_len: int
    ) -> List[Tuple[int, int]]:
        """Build edges based on semantic similarity threshold."""
        edges = []
        
        # Use first batch item for edge computation
        # [T, D]
        state = semantic_state[0] if semantic_state.dim() == 3 else semantic_state
        
        # Normalize for cosine similarity
        state_norm = F.normalize(state, dim=-1)
        
        # Compute pairwise similarity [T, T]
        similarity = torch.mm(state_norm, state_norm.t())
        
        # Find pairs above threshold (excluding diagonal and sequential)
        for i in range(seq_len):
            for j in range(i + 2, seq_len):  # Skip i, i+1
                if similarity[i, j].item() > self.semantic_threshold:
                    edges.append((i, j))
                    if self.bidirectional:
                        edges.append((j, i))
        
        return edges
    
    def _build_shortcuts(self, seq_len: int) -> List[Tuple[int, int]]:
        """Build random small-world shortcuts."""
        edges = []
        
        for i in range(seq_len):
            for j in range(i + 2, seq_len):  # Skip adjacent
                if random.random() < self.shortcut_prob:
                    edges.append((i, j))
                    if self.bidirectional:
                        edges.append((j, i))
        
        return edges
    
    def update_config(
        self,
        enable_semantic: Optional[bool] = None,
        enable_shortcuts: Optional[bool] = None,
        semantic_threshold: Optional[float] = None,
        semantic_k: Optional[int] = None,
        semantic_method: Optional[str] = None,
        shortcut_prob: Optional[float] = None
    ):
        """Update graph builder configuration."""
        if enable_semantic is not None:
            self.enable_semantic = enable_semantic
        if enable_shortcuts is not None:
            self.enable_shortcuts = enable_shortcuts
        if semantic_threshold is not None:
            self.semantic_threshold = semantic_threshold
        if semantic_k is not None:
            self.semantic_k = semantic_k
        if semantic_method is not None:
            self.semantic_method = semantic_method
        if shortcut_prob is not None:
            self.shortcut_prob = shortcut_prob
