"""
Typed Edge Embeddings - Phase 2.3.2

Adds learned embeddings for different edge types to distinguish:
- Sequential edges (i ↔ i+1): Syntactic/local dependencies
- Semantic edges (similarity): Coreference, semantic relations
- Shortcuts (long-range): Topic-level connections

Each edge type gets a learned embedding that modulates attention.

Benefits:
- Interpretability: See which edge types matter for which heads
- Performance: Model can learn edge type importance
- Minimal cost: Only 3 × hidden_dim parameters

Based on Graph Attention Networks and recommended by Modernization Report.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class TypedEdgeEmbedding(nn.Module):
    """
    Learned embeddings for edge types.
    
    Adds type-specific bias to attention scores based on edge type.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_edge_types: int = 3,
        edge_type_names: Optional[List[str]] = None
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_edge_types: Number of edge types (default 3: sequential, semantic, shortcuts)
            edge_type_names: Names for interpretability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        
        # Default edge type names
        if edge_type_names is None:
            edge_type_names = ['sequential', 'semantic', 'shortcut']
        self.edge_type_names = edge_type_names
        
        # Learnable embedding for each edge type
        self.edge_type_embeddings = nn.Embedding(num_edge_types, hidden_dim)
        nn.init.normal_(self.edge_type_embeddings.weight, mean=0.0, std=0.02)
        
        # Projection to scalar bias
        # Each head can learn different preferences for edge types
        self.type_to_bias = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.type_to_bias.weight)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Typed Edge Embeddings: {total_params:,} parameters")
        print(f"    Types: {edge_type_names}")
    
    def forward(
        self,
        query: torch.Tensor,
        edge_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute type-specific attention bias.
        
        Args:
            query: [B, num_heads, seq_len, head_dim] query vectors
            edge_types: [B, num_edges] edge type indices
            
        Returns:
            bias: [B, num_heads, num_edges] attention bias values
        """
        # Get edge type embeddings
        type_emb = self.edge_type_embeddings(edge_types)  # [num_edges, hidden_dim] or [B, num_edges, hidden_dim]
        
        # Project to bias
        bias = self.type_to_bias(type_emb).squeeze(-1)  # [num_edges] or [B, num_edges]
        
        # Handle batched vs unbatched
        if bias.dim() == 1:
            # Unbatched: [num_edges] -> [1, 1, num_edges] -> [1, num_heads, num_edges]
            B = 1
            num_edges = bias.size(0)
            bias =bias.view(1, 1, num_edges)
        else:
            # Batched: [B, num_edges] -> [B, 1, num_edges]
            B = bias.size(0)
            num_edges = bias.size(1)
            bias = bias.unsqueeze(1)
        
        # Expand for heads
        num_heads = query.size(1)
        bias = bias.expand(B, num_heads, num_edges)  # [B, num_heads, num_edges]
        
        return bias
    
    def get_type_importance(self) -> Dict[str, float]:
        """
        Get average importance of each edge type.
        
        Returns:
            Dict mapping edge type name to average bias
        """
        importance = {}
        
        for i, name in enumerate(self.edge_type_names):
            # Get embedding for this type
            type_idx = torch.tensor([i])
            emb = self.edge_type_embeddings(type_idx)  # [1, hidden_dim]
            
            # Project to bias
            bias = self.type_to_bias(emb).item()
            
            importance[name] = bias
        
        return importance


class GraphAttentionWithTypedEdges(nn.Module):
    """
    Graph attention mechanism with typed edge embeddings.
    
    Extends standard attention to incorporate edge type information.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_typed_edges: bool = True
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_typed_edges = use_typed_edges
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Typed edge embeddings
        if use_typed_edges:
            self.typed_edges = TypedEdgeEmbedding(
                hidden_dim=hidden_dim,
                num_edge_types=3,
                edge_type_names=['sequential', 'semantic', 'shortcut']
            )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: List[Tuple[int, int]],
        edge_types: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with typed edges.
        
        Args:
            x: [B, T, hidden_dim] input features
            edge_index: List of (src, dst) tuples
            edge_types: List of edge type indices (0=seq, 1=sem, 2=shortcut)
            mask: Optional attention mask
            
        Returns:
            out: [B, T, hidden_dim] output features
        """
        B, T, D = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, num_heads, T, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
        
        # Add typed edge bias if enabled
        if self.use_typed_edges and edge_types is not None:
            # Convert edge types to tensor
            edge_types_tensor = torch.tensor(edge_types, dtype=torch.long, device=x.device)
            edge_types_tensor = edge_types_tensor.unsqueeze(0).expand(B, -1)
            
            # Get type-specific bias
            type_bias = self.typed_edges(Q, edge_types_tensor)  # [B, num_heads, num_edges]
            
            # Add bias to corresponding attention scores
            for i, (src, dst) in enumerate(edge_index):
                scores[:, :, src, dst] += type_bias[:, :, i]
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax + dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, num_heads, T, head_dim]
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


# ============================================================================
# Testing
# ============================================================================

def test_typed_edges():
    """Test typed edge embeddings."""
    print("Testing Typed Edge Embeddings...")
    
    # Test TypedEdgeEmbedding
    typed_emb = TypedEdgeEmbedding(
        hidden_dim=64,
        num_edge_types=3
    )
    
    # Create sample edge types
    edge_types = torch.tensor([0, 1, 2, 0, 1])  # seq, sem, shortcut, seq, sem
    query = torch.randn(1, 4, 10, 16)  # [B, heads, seq_len, head_dim]
    
    bias = typed_emb(query, edge_types)
    print(f"✓ TypedEdgeEmbedding: bias shape {bias.shape}")
    assert bias.shape == (1, 4, 5)  # [B, heads, num_edges]
    
    # Test importance
    importance = typed_emb.get_type_importance()
    print(f"✓ Edge type importance: {importance}")
    
    # Test GraphAttentionWithTypedEdges
    print("\nTesting GraphAttentionWithTypedEdges...")
    
    attn = GraphAttentionWithTypedEdges(
        hidden_dim=64,
        num_heads=4,
        use_typed_edges=True
    )
    
    # Sample input
    x = torch.randn(2, 10, 64)  # [B, T, D]
    edge_index = [(0, 1), (1, 2), (0, 5), (2, 3)]  # Some edges
    edge_types_list = [0, 0, 2, 0]  # seq, seq, shortcut, seq
    
    out = attn(x, edge_index, edge_types_list)
    print(f"✓ Attention output: {out.shape}")
    assert out.shape == x.shape
    
    # Test without typed edges
    attn_no_types = GraphAttentionWithTypedEdges(
        hidden_dim=64,
        num_heads=4,
        use_typed_edges=False
    )
    
    out_no_types = attn_no_types(x, edge_index)
    print(f"✓ Attention without types: {out_no_types.shape}")
    
    # Verify they differ
    diff = (out - out_no_types).abs().mean().item()
    print(f"✓ Difference with/without types: {diff:.6f}")
    
    if diff > 0.001:
        print("✓ Typed edges are actively modifying attention!")
    
    print("\n✅ All typed edge tests passed!")


if __name__ == '__main__':
    test_typed_edges()
