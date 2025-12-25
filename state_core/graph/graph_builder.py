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
        bidirectional: bool = True,
        use_mutual_knn: bool = True,  # PHASE 1: Mutual k-NN filtering
        streaming_topk: bool = True,  # PHASE 1: Streaming Top-K (avoid T×T matrix)
        semantic_blocks: Optional[List[str]] = None,  # PHASE 2: Which MU blocks to use for similarity
        use_ema: bool = False,  # PHASE 2: Use EMA for stable graph topology
        ema_decay: float = 0.99  # PHASE 2: EMA decay rate (higher = more stable)
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
        self.use_mutual_knn = use_mutual_knn  # PHASE 1
        self.streaming_topk = streaming_topk  # PHASE 1
        self.semantic_blocks = semantic_blocks if semantic_blocks else ['I', 'R2', 'K']  # PHASE 2: Default to I, R2, K blocks
        
        # PHASE 2: EMA for graph stability
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.mu_ema = None  # Will be initialized on first forward pass
    
    def _update_ema(self, mu_current: torch.Tensor) -> torch.Tensor:
        """
        PHASE 2: Update and return EMA of MU embeddings.
        
        This prevents graph "flickering" during training by using smoothed embeddings.
        The graph topology evolves slowly, giving attention layers a stable target.
        
        Args:
            mu_current: [B, T, D] current MU embeddings
        
        Returns:
            mu_stable: [B, T, D] EMA-smoothed embeddings for graph construction
        """
        if not self.use_ema:
            return mu_current
        
        # Initialize EMA state on first call
        if self.mu_ema is None:
            self.mu_ema = mu_current.detach().clone()
            return mu_current
        
        # Update EMA: ema = decay * ema_prev + (1 - decay) * current
        with torch.no_grad():
            self.mu_ema = self.ema_decay * self.mu_ema + (1 - self.ema_decay) * mu_current.detach()
        
        return self.mu_ema
    
    def _select_semantic_blocks(self, mu_state: torch.Tensor) -> torch.Tensor:
        """
        PHASE 2: Extract specific MU blocks for similarity computation.
        
        MU structure: 8×8 matrix = 16 blocks of 4D each = 64D total
        Blocks: I, D, R1, R2, K, M, T, P, S, C, N, X, E, F, A, Z
        
        Common selections:
        - ['I', 'R2', 'K']: Identity + Relations + Keys (12D) - semantic core
        - ['I', 'D']: Identity + Domain (8D) - very fast
        - All blocks: Use full 64D (original behavior)
        
        Args:
            mu_state: [B, T, 64] MU semantic state
        
        Returns:
            [B, T, block_dim] Selected blocks concatenated
        """
        if len(self.semantic_blocks) == 16:  # All blocks
            return mu_state
        
        # Map block names to indices (each block is 4D)
        block_map = {
            'I': 0, 'D': 1, 'R1': 2, 'R2': 3,
            'K': 4, 'M': 5, 'T': 6, 'P': 7,
            'S': 8, 'C': 9, 'N': 10, 'X': 11,
            'E': 12, 'F': 13, 'A': 14, 'Z': 15
        }
        
        # Select block slices
        selected_slices = []
        for block_name in self.semantic_blocks:
            if block_name in block_map:
                block_idx = block_map[block_name]
                start = block_idx * 4
                end = start + 4
                selected_slices.append(mu_state[..., start:end])
        
        # Concatenate selected blocks
        return torch.cat(selected_slices, dim=-1)
    
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
        
        # Semantic edges based on similarity
        provenance = []
        if self.enable_semantic and semantic_state is not None:
            # PHASE 2: Apply EMA smoothing for stable graph topology
            semantic_state_stable = self._update_ema(semantic_state)
            
            if self.semantic_method == 'topk':
                semantic_edges, sem_prov = self._build_semantic_edges_topk(semantic_state_stable, seq_len)
                if sem_prov:
                    provenance.extend(sem_prov)
            else:
                semantic_edges = self._build_semantic_edges_threshold(semantic_state_stable, seq_len)
            
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
            'batch_size': batch_size,
            'provenance': provenance if provenance else None  # PHASE 2.2: Include provenance
        }
    
    
    def _build_semantic_edges_topk(
        self,
        semantic_state: torch.Tensor,
        seq_len: int
    ) -> List[Tuple[int, int]]:
        """
        Build top-K semantic edges per token.
        
        PHASE 1: Uses streaming Top-K to avoid materializing full T×T matrix.
        Memory: O(T×K) instead of O(T²)
        """
        if self.streaming_topk:
            return self._streaming_topk(semantic_state, seq_len)
        else:
            return self._materialized_topk(semantic_state, seq_len)
    
    def _streaming_topk(
        self,
        semantic_state: torch.Tensor,
        seq_len: int,
        track_provenance: bool = True  # NEW: Track block contributions
    ) -> Tuple[List[Tuple[int, int]], Optional[List[Dict]]]:
        """
        PHASE 1: Streaming Top-K - compute similarities row-by-row.
        Avoids creating full T×T matrix.
        
        PHASE 2.2: Now tracks provenance (which blocks contribute to each edge)
        """
        edges = []
        provenance = [] if track_provenance else None
        
        # Use first batch item
        state = semantic_state[0] if semantic_state.dim() == 3 else semantic_state
        
        if track_provenance:
            # PHASE 2.2: Extract individual block states for provenance
            from state_core.adapters.mu_adapter import SemanticBlockLayout
            
            block_states = {}
            for block_name in self.semantic_blocks:
                # Get indices for this block in 64D vector
                indices = SemanticBlockLayout.get_flat_indices(block_name)
                block_states[block_name] = state[:, indices]  # [T, 4] for each 2×2 block
            
            # Normalize each block separately
            block_states_norm = {}
            for block_name, block_state in block_states.items():
                block_states_norm[block_name] = F.normalize(block_state, dim=-1)
        
        # PHASE 2: Select semantic blocks for combined similarity (default: I, R2, K = 12D)
        state = self._select_semantic_blocks(state.unsqueeze(0) if state.dim() == 2 else state)
        state = state[0] if state.dim() == 3 else state  # Back to [T, d]
        
        # Normalize once
        state_norm = F.normalize(state, dim=-1)  # [T, reduced_d]
        
        # For each token, compute similarities with all others (one row only!)
        for i in range(seq_len):
            query = state_norm[i]  # [d]
            
            # Compute similarities with all others (one row only!)
            sims = (state_norm @ query)  # [T] - NOT [T, T]
            
            # PHASE 2.2: Compute per-block similarities if tracking provenance
            block_sims = None
            if track_provenance:
                block_sims = {}
                for block_name, block_norm in block_states_norm.items():
                    block_query = block_norm[i]  # [4]
                    block_sims[block_name] = (block_norm @ block_query)  # [T]
            
            # Mask self and adjacent
            sims[i] = -float('inf')
            if i > 0:
                sims[i-1] = -float('inf')
            if i < seq_len - 1:
                sims[i+1] = -float('inf')
            
            # Top-K
            k = min(self.semantic_k, seq_len - 3)
            if k > 0:
                top_k_sim, top_k_idx = sims.topk(k)
                
                for j_idx, sim_val in zip(top_k_idx, top_k_sim):
                    j = j_idx.item()
                    
                    # Optional minimum threshold
                    if self.semantic_threshold > 0 and sim_val < self.semantic_threshold:
                        continue
                    
                    # Add edge (only once per pair)
                    if j > i:
                        edges.append((i, j))
                        if self.bidirectional:
                            edges.append((j, i))
                        
                        # PHASE 2.2: Track provenance
                        if track_provenance and block_sims is not None:
                            prov_data = {
                                'edge': (i, j),
                                'combined_similarity': sim_val.item(),
                            }
                            
                            # Add per-block contributions
                            for block_name in self.semantic_blocks:
                                prov_data[f'{block_name}_similarity'] = block_sims[block_name][j].item()
                            
                            provenance.append(prov_data)
        
        # PHASE 1: Apply mutual k-NN filtering
        if self.use_mutual_knn:
            edges = self._mutual_knn_filter(edges)
            # Note: provenance entries may refer to filtered edges
            # This is OK - we track all candidates
        
        return edges, provenance
    
    def _materialized_topk(
        self,
        semantic_state: torch.Tensor,
        seq_len: int
    ) -> List[Tuple[int, int]]:
        """
        Original Top-K implementation (for comparison).
        Creates full T×T similarity matrix.
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
        """
        Build Fibonacci-distance shortcuts for better long-range connectivity.
        
        PHASE 2.4: Replaces random shortcuts with structured Fibonacci pattern.
        For each token i, connect to i±fib(k) for k in [3, 4, 5, ...]
        Fibonacci: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144...
        
        Benefits:
        - More structured than random
        - Better coverage of distance scales
        - Natural spacing (fibonacci growth ~1.618)
        - Same edge count as random on average
        """
        edges = []
        
        # Generate Fibonacci numbers up to seq_len
        # Start from fib(3)=2 to avoid overlapping with sequential edges
        fibs = [2, 3]
        while fibs[-1] < seq_len:
            fibs.append(fibs[-1] + fibs[-2])
        
        # Determine how many Fibonacci connections per token
        # Use shortcut_prob to control density: prob=0.05 → ~1-2 connections per token
        max_fibs_per_token = max(1, int(self.shortcut_prob * 20))  # 0.05 * 20 = 1
        
        # For each position
        for i in range(seq_len):
            fib_count = 0
            for fib_dist in fibs:
                if fib_count >= max_fibs_per_token:
                    break
                if fib_dist >= seq_len:
                    break
                
                # Forward connection
                if i + fib_dist < seq_len:
                    edges.append((i, i + fib_dist))
                    if self.bidirectional:
                        edges.append((i + fib_dist, i))
                    fib_count += 1
                
                # Backward connection (alternating to maintain balance)
                elif i - fib_dist >= 0 and fib_count < max_fibs_per_token:
                    edges.append((i, i - fib_dist))
                    if self.bidirectional:
                        edges.append((i - fib_dist, i))
                    fib_count += 1
        
        return edges
    
    def _mutual_knn_filter(self, edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        PHASE 1: Mutual k-NN filtering.
        Keep only bidirectional edges (i→j AND j→i exist).
        
        Benefits:
        - Reduces hub tokens (high-degree nodes)
        - Higher precision edges
        - Typically 20-30% edge reduction
        """
        edge_set = set(edges)
        mutual_edges = []
        
        for (i, j) in edges:
            # Check if reverse edge also exists
            if (j, i) in edge_set and i < j:  # Avoid duplicates
                mutual_edges.append((i, j))
                if self.bidirectional:
                    mutual_edges.append((j, i))
        
        return mutual_edges
    
    def update_config(
        self,
        enable_semantic: Optional[bool] = None,
        enable_shortcuts: Optional[bool] = None,
        semantic_threshold: Optional[float] = None,
        semantic_k: Optional[int] = None,
        semantic_method: Optional[str] = None,
        shortcut_prob: Optional[float] = None,
        use_mutual_knn: Optional[bool] = None,  # PHASE 1
        streaming_topk: Optional[bool] = None,  # PHASE 1
        semantic_blocks: Optional[List[str]] = None,  # PHASE 2
        use_ema: Optional[bool] = None,  # PHASE 2
        ema_decay: Optional[float] = None  # PHASE 2
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
        if use_mutual_knn is not None:  # PHASE 1
            self.use_mutual_knn = use_mutual_knn
        if streaming_topk is not None:  # PHASE 1
            self.streaming_topk = streaming_topk
        if semantic_blocks is not None:  # PHASE 2
            self.semantic_blocks = semantic_blocks
        if use_ema is not None:  # PHASE 2
            self.use_ema = use_ema
        if ema_decay is not None:  # PHASE 2
            self.ema_decay = ema_decay
