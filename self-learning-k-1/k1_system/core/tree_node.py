"""
TreeNode: Single node in the K-1 hierarchical tree.

Each node is a small transformer block that processes its input.
Nodes are organized in a tree structure: Root → Nodes → Agents → Sub-Agents.
"""

import torch
import torch.nn as nn
from typing import Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    pass  # Avoid circular imports


class K1_Node(nn.Module):
    """
    Single node in the hierarchical tree.
    Each node is a small transformer block that processes its input.
    
    Attributes:
        embed_dim: Embedding dimension
        node_id: Unique identifier for this node
        level: Depth level in tree (0=root, 1=nodes, 2=agents, 3=sub-agents)
        is_leaf: Whether this node has no children
        child_nodes: List of child nodes
        last_updated_step: Step when this node was last updated (for trust cooldown)
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        ff_dim: int, 
        num_heads: int = 4, 
        dropout: float = 0.1
    ):
        """
        Initialize a TreeNode.
        
        Args:
            embed_dim: Embedding dimension
            ff_dim: Feed-forward hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Self-attention (with Flash Attention if available)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(embed_dim)
        
        # Router to children (if not leaf)
        self.router = None  # Set by parent
        self.child_nodes = nn.ModuleList()  # CRITICAL: ModuleList for device transfer
        self.is_leaf = True
        
        # Tracking
        self.node_id: Optional[int] = None
        self.level: int = 0
        self.activation_count: int = 0
        self.gradient_norm: float = 0.0
        self.last_updated_step: int = -1000  # For trust cooldown
        self.update_count: int = 0  # Track total updates for analysis
        
        # Specialization tracking (for interpretability)
        self.token_counts = {}  # {token_id: count} - track which tokens this node handles
        self.error_sum = 0.0  # Sum of errors when this node was culprit
        self.specialization_label = None  # Human-readable label (e.g., "nouns", "verbs")
        
        # Domain specialization tracking
        self.domain_counts = {}  # {domain_name: count} - track which domains this node handles
    
    def record_tokens(self, token_ids: torch.Tensor):
        """
        Record which tokens this node processed when it was the culprit.
        
        Args:
            token_ids: Tensor of token IDs that caused errors
        """
        for tid in token_ids.flatten().tolist():
            self.token_counts[tid] = self.token_counts.get(tid, 0) + 1
    
    def record_domain(self, domain_name: str):
        """
        Record which domain this node handled when it was the culprit.
        
        Args:
            domain_name: Name of the domain (e.g., 'wikitext', 'code', 'scientific')
        """
        self.domain_counts[domain_name] = self.domain_counts.get(domain_name, 0) + 1
    
    def get_primary_domain(self) -> tuple:
        """
        Get the domain this node handles most often.
        
        Returns:
            Tuple of (domain_name, count, percentage) or (None, 0, 0) if no data
        """
        if not self.domain_counts:
            return (None, 0, 0.0)
        
        total = sum(self.domain_counts.values())
        top_domain = max(self.domain_counts.items(), key=lambda x: x[1])
        return (top_domain[0], top_domain[1], top_domain[1] / total * 100)
    
    def get_domain_distribution(self) -> dict:
        """
        Get the distribution of domains this node handles.
        
        Returns:
            Dictionary with {domain: percentage}
        """
        if not self.domain_counts:
            return {}
        
        total = sum(self.domain_counts.values())
        return {domain: count / total * 100 for domain, count in self.domain_counts.items()}
    
    def get_top_tokens(self, n: int = 10) -> list:
        """
        Get the top N tokens this node handles most often.
        
        Args:
            n: Number of top tokens to return
            
        Returns:
            List of (token_id, count) tuples
        """
        sorted_tokens = sorted(self.token_counts.items(), key=lambda x: -x[1])
        return sorted_tokens[:n]
    
    def get_specialization_score(self) -> dict:
        """
        Get a summary of this node's specialization.
        
        Returns:
            Dictionary with specialization metrics
        """
        total = sum(self.token_counts.values())
        primary_domain = self.get_primary_domain()
        return {
            'node_id': self.node_id,
            'level': self.level,
            'update_count': self.update_count,
            'unique_tokens': len(self.token_counts),
            'total_tokens': total,
            'top_tokens': self.get_top_tokens(5),
            'primary_domain': primary_domain[0],
            'domain_confidence': primary_domain[2],
            'domain_distribution': self.get_domain_distribution(),
            'label': self.specialization_label
        }
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process input through this node.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Processed tensor [batch, seq_len, embed_dim]
        """
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = self.attn_norm(x + attn_out)
        
        # FFN with residual
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)
        
        # NOTE: activation_count removed to prevent torch.compile recompilation
        return x
    
    def add_child(self, child: 'TreeNode'):
        """
        Add a child node.
        
        Args:
            child: Child TreeNode to add
        """
        self.child_nodes.append(child)
        self.is_leaf = False
        child.level = self.level + 1
    
    def get_gradient_norm(self) -> float:
        """
        Compute gradient norm for this node's parameters.
        
        OPTIMIZED: Keeps computation on GPU, returns as Python float
        but computation happens in single batched operation.
        
        Returns:
            L2 norm of all gradients in this node
        """
        # Gather all gradients that exist
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        
        if not grads:
            self.gradient_norm = 0.0
            return 0.0
        
        # Efficient: stack and compute single norm (stays on GPU longer)
        norm_tensor = torch.stack([g.norm() for g in grads]).norm()
        self.gradient_norm = norm_tensor.item()  # Single sync point
        return self.gradient_norm
    
    def __repr__(self) -> str:
        return (
            f"K1_Node(id={self.node_id}, level={self.level}, "
            f"is_leaf={self.is_leaf}, children={len(self.child_nodes)})"
        )
