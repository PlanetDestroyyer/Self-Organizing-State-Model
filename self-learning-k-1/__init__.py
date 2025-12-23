"""
Self-Learning K-1 Module

Hierarchical tree with gradient-based credit assignment.
"""

import sys
from pathlib import Path

# Add k1_system to path
_k1_path = Path(__file__).parent / "k1_system"
if str(_k1_path) not in sys.path:
    sys.path.insert(0, str(_k1_path))

# Core exports (all prefixed with K1_)
from k1_system.core.tree import K1_Tree
from k1_system.core.tree_node import K1_Node
from data.loader import K1_DataLoader

__all__ = [
    'K1_Tree',
    'K1_Node',
    'K1_DataLoader',
]
