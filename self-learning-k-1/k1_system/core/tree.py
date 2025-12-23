import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional

from .tree_node import K1_Node

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class K1_Tree(nn.Module):
    """
    Hierarchical tree of nodes for language modeling.
    
    Structure:
    - Shared embedding layer
    - Tree of K1_Node modules (depth configurable)
    - Shared output projection
    
    Key innovation: Path-based gradient updates with improved responsibility signal.
    
    Attributes:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        tree_depth: Number of levels in tree
        branching_factor: Children per node at each level
        all_nodes: List of all K1_Node instances
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        ff_dim: int = 256,
        num_heads: int = 4,
        tree_depth: int = 4,          # Root + Nodes + Agents + Sub-Agents
        branching_factor=None,        # [4, 3, 2] or single int
        max_seq_len: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize HierarchicalTree.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            ff_dim: Feed-forward hidden dimension
            num_heads: Number of attention heads
            tree_depth: Depth of tree (e.g., 4 = Root→Nodes→Agents→SubAgents)
            branching_factor: Children per node, e.g. [4, 3, 2]
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.tree_depth = tree_depth
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Gradient norm cache (optimization)
        self._grad_norm_cache: dict = {}

        # Support variable branching: [4 nodes, 3 agents, 2 sub-agents]
        if branching_factor is None:
            branching_factor = [4, 3, 2]
        elif isinstance(branching_factor, int):
            branching_factor = [branching_factor] * (tree_depth - 1)

        self.branching_factor = branching_factor
        
        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.02)
        
        # Build hierarchical tree
        self.root = self._build_tree(depth=0, max_depth=tree_depth, node_id=0)
        
        # Collect all nodes for easy access
        self.all_nodes: List[K1_Node] = []
        self._collect_nodes(self.root)
        
        # Build node-to-index mapping for O(1) lookup (optimization)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.all_nodes)}
        
        # Pre-group parameters by node for vectorized gradient computation
        self.node_params = [list(node.parameters()) for node in self.all_nodes]
        
        # Shared output projection
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Print structure
        print(f"Created HierarchicalTree:")
        print(f"  Vocab: {vocab_size}, Embed: {embed_dim}")
        print(f"  Tree depth: {tree_depth}, Branching: {branching_factor}")
        print(f"  Total nodes: {len(self.all_nodes)}")
        print(f"  Leaf nodes: {sum(1 for n in self.all_nodes if n.is_leaf)}")
    
    def _build_tree(self, depth: int, max_depth: int, node_id: int) -> K1_Node:
        """
        Recursively build the tree with variable branching.

        Structure:
          Root (depth=0, hidden)
            → 4 Nodes (depth=1)
              → 3 Agents per Node (depth=2)
                → 2 Sub-Agents per Agent (depth=3)
        """
        node = K1_Node(self.embed_dim, self.ff_dim, self.num_heads, self.dropout)
        node.node_id = node_id
        node.level = depth

        if depth < max_depth - 1:
            # Get branching factor for this level
            if isinstance(self.branching_factor, list):
                num_children = (
                    self.branching_factor[depth] 
                    if depth < len(self.branching_factor) 
                    else self.branching_factor[-1]
                )
            else:
                num_children = self.branching_factor

            # Create children
            for i in range(num_children):
                child_id = (
                    len(self.all_nodes) 
                    if hasattr(self, 'all_nodes') 
                    else node_id * 10 + i + 1
                )
                child = self._build_tree(depth + 1, max_depth, child_id)
                node.add_child(child)

        return node
    
    def _collect_nodes(self, node: K1_Node):
        """Collect all nodes into a list."""
        self.all_nodes.append(node)
        for child in node.child_nodes:
            self._collect_nodes(child)
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the tree using LEVEL-WISE batched processing.
        
        This is much faster than recursive because all nodes at same level
        are processed in parallel.
        
        NOTE: Only returns logits (tensor) for DataParallel compatibility.
        Use get_last_path() to access the path after forward.
        
        Args:
            x: Input token indices [batch, seq_len]
        
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        # Embed tokens
        h = self.embedding(x)  # [batch, seq, embed]
        h = h + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Create causal mask ONCE
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Track path (stored internally for DataParallel compatibility)
        self._last_path = []
        
        # Level 0: Root
        h = self.root(h, mask)
        self._last_path.append(self.root)
        
        # Process remaining levels in PARALLEL (OPTIMIZED)
        current_nodes = list(self.root.child_nodes)
        
        while current_nodes:
            # OPTIMIZATION: Pre-allocate instead of creating lists
            num_nodes = len(current_nodes)
            next_level_nodes = []
            
            if num_nodes == 1:
                # Fast path for single node
                h = current_nodes[0](h, mask)
                self._last_path.append(current_nodes[0])
                next_level_nodes.extend(current_nodes[0].child_nodes)
            else:
                # Multiple nodes: accumulate outputs efficiently
                level_output = torch.zeros_like(h)
                
                for node in current_nodes:
                    out = node(h, mask)
                    level_output += out  # In-place accumulation
                    self._last_path.append(node)
                    next_level_nodes.extend(node.child_nodes)
                
                # In-place mean
                h = level_output / num_nodes
            
            current_nodes = next_level_nodes
        
        # Output projection
        h = self.output_norm(h)
        logits = self.output_proj(h)
        
        return logits
    
    def get_last_path(self) -> List[K1_Node]:
        """Get the path from the last forward pass."""
        return getattr(self, '_last_path', self.all_nodes)
    
    # ========================================
    # Responsibility Signal Methods (OPTIMIZED)
    # ========================================
    
    def cache_all_gradient_norms(self):
        """Cache gradient norms for all nodes (call once per step)."""
        self._grad_norm_cache = {}
        for node in self.all_nodes:
            total = 0.0
            for p in node.parameters():
                if p.grad is not None:
                    total += p.grad.norm().item()
            # NaN protection
            if math.isnan(total) or math.isinf(total):
                total = 0.0001
            self._grad_norm_cache[node.node_id] = total
    
    def fast_hierarchical_step(self, loss_tensor: torch.Tensor, current_step: int):
        """
        Hierarchical error attribution - trace errors through the tree.
        
        Core K-1 Innovation:
        1. Compute gradients for all nodes
        2. Find which node has highest gradient (most responsible for error)
        3. Apply proportional updates: Culprit=100%, Parent=15%, Root=5%
        
        This provides INTERPRETABILITY - know exactly which node caused the error.
        
        Args:
            loss_tensor: The loss tensor (still on GPU, don't call .item())
            current_step: Current training step
        """
        # Collect ALL gradients from ALL nodes in single flat list
        all_grads = []
        grad_to_node = []  # Maps gradient index to node index
        
        for node_idx, node_params in enumerate(self.node_params):
            for p in node_params:
                if p.grad is not None:
                    all_grads.append(p.grad)
                    grad_to_node.append(node_idx)
        
        if not all_grads:
            return
        
        # Compute ALL gradient norms in SINGLE batched operation (GPU)
        all_grad_norms = torch.stack([g.norm() for g in all_grads])
        
        # Sum gradient norms per node using scatter_add (vectorized)
        num_nodes = len(self.all_nodes)
        grad_to_node_tensor = torch.tensor(grad_to_node, device=loss_tensor.device)
        grad_tensor = torch.zeros(num_nodes, device=loss_tensor.device)
        grad_tensor.scatter_add_(0, grad_to_node_tensor, all_grad_norms)
        
        # Cache for interpretability logging
        self._grad_tensor = grad_tensor
        
        # ========================================
        # HIERARCHICAL ERROR ATTRIBUTION
        # Drill down until we reach a leaf: Root → Node → Agent → Sub-Agent
        # ========================================
        
        path_indices = [0]  # Start with root
        current_node = self.root
        
        # Keep drilling down until we hit a leaf
        while current_node.child_nodes:
            child_indices = [self.node_to_idx[c] for c in current_node.child_nodes]
            child_grads = grad_tensor[torch.tensor(child_indices, device=loss_tensor.device)]
            best_child_local = child_grads.argmax().item()
            best_child_idx = child_indices[best_child_local]
            
            path_indices.append(best_child_idx)
            current_node = self.all_nodes[best_child_idx]
        
        # Store for interpretability output
        self._error_path = path_indices
        self._path_gradients = [grad_tensor[i].item() for i in path_indices]
        
        # ========================================
        # DYNAMIC GRADIENT-PROPORTIONAL SCALING
        # Higher gradient = higher responsibility = higher update scale
        # ========================================
        num_nodes = len(self.all_nodes)
        scales = torch.zeros(num_nodes, device=loss_tensor.device)
        
        # Get gradients for path nodes
        path_grads = torch.tensor([grad_tensor[i].item() for i in path_indices], device=loss_tensor.device)
        
        # Normalize to get proportional responsibility
        # Culprit (last) still gets at least 50%, others proportional
        if path_grads.sum() > 0:
            # Softmax-style proportional scaling
            normalized_grads = path_grads / path_grads.sum()
            
            for i, node_idx in enumerate(path_indices):
                if i == len(path_indices) - 1:
                    # Culprit: 50% base + proportional boost (max 100%)
                    scales[node_idx] = min(1.0, 0.5 + normalized_grads[i].item())
                else:
                    # Others: proportional to their gradient contribution (capped at 30%)
                    scales[node_idx] = min(0.3, normalized_grads[i].item())
        else:
            # Fallback to fixed scales if no gradients
            for i, node_idx in enumerate(path_indices):
                if i == len(path_indices) - 1:
                    scales[node_idx] = 1.0
                elif i == 0:
                    scales[node_idx] = 0.05
                else:
                    scales[node_idx] = 0.1
        
        # Store dynamic scales for interpretability
        self._dynamic_scales = {path_indices[i]: scales[path_indices[i]].item() for i in range(len(path_indices))}
        
        # Apply scales to gradients
        for grad_idx, node_idx in enumerate(grad_to_node):
            scale = scales[node_idx]
            if scale > 0:
                all_grads[grad_idx].mul_(scale)
        
        # Store for tracking
        self._last_scales_indices = path_indices
        self._last_scales_tensor = scales
    
    def get_error_attribution(self) -> dict:
        """
        Get interpretable error attribution for the last step.
        
        Returns:
            Dictionary with error path and gradient information.
        """
        if not hasattr(self, '_error_path'):
            return {}
        
        path_info = []
        for i, node_idx in enumerate(self._error_path):
            node = self.all_nodes[node_idx]
            role = ["Root", "Node", "Culprit"][min(i, 2)]
            scale = [5, 15, 100][min(i, 2)]
            path_info.append({
                'node_id': node_idx,
                'role': role,
                'gradient': self._path_gradients[i] if hasattr(self, '_path_gradients') else 0,
                'update_scale': f"{scale}%"
            })
        
        return {
            'error_path': path_info,
            'culprit_node': self._error_path[-1],
            'total_nodes': len(self.all_nodes),
            'nodes_updated': len(self._error_path)
        }

    
    def _get_node_gradient_norm(self, node: K1_Node) -> float:
        """Get gradient norm for a specific node (uses cache if available)."""
        if node.node_id in self._grad_norm_cache:
            return self._grad_norm_cache[node.node_id]
        
        # Fallback: compute directly
        total = 0.0
        for p in node.parameters():
            if p.grad is not None:
                total += p.grad.norm().item()
        # NaN protection
        if math.isnan(total) or math.isinf(total):
            total = 0.0001
        return total



    # ========================================
    # Utility Methods
    # ========================================

    def get_path_gradient_norms(self) -> dict:
        """Get gradient norms for all nodes."""
        return {node.node_id: node.get_gradient_norm() for node in self.all_nodes}
    
    def get_node_by_id(self, node_id: int) -> Optional[K1_Node]:
        """Get node by ID."""
        for node in self.all_nodes:
            if node.node_id == node_id:
                return node
        return None

    def print_responsibility_tree(
        self, 
        grad_norms: dict, 
        scales: dict, 
        node: K1_Node = None, 
        level: int = 0
    ):
        """Print hierarchical responsibility visualization."""
        if node is None:
            node = self.root

        indent = "  " * level
        node_id = node.node_id
        grad = grad_norms.get(node_id, 0.0)
        scale = scales.get(node_id, 0.0)

        # Status icon
        if scale >= 0.8:
            icon = "🚨"  # Culprit
        elif scale > 0:
            icon = "⚠️"   # Parent/Manager
        else:
            icon = "✓"

        # Role name
        roles = {0: "Root   ", 1: "Node   ", 2: "Agent  "}
        role = roles.get(level, "SubAgent")

        print(f"{indent}{icon} {role} {node_id}: grad={grad:.3f}, update={scale*100:3.0f}%")

        for child in node.child_nodes:
            self.print_responsibility_tree(grad_norms, scales, child, level + 1)
