"""
K-1 Adapter - Wraps K-1 hierarchical learning.

Intercepts gradients after backward pass.
Lets K-1 decide which parameters update and by how much.
Does NOT alter K-1 internals.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

# Add K-1 repo to path
K1_PATH = Path(__file__).parent.parent.parent / "self-learning-k-1"
if str(K1_PATH) not in sys.path:
    sys.path.insert(0, str(K1_PATH))


class K1Adapter:
    """
    Adapter for K-1 hierarchical learning.
    
    Wraps K-1's gradient attribution and selective update logic.
    Call after loss.backward() to apply hierarchical credit assignment.
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_hierarchical_tree: bool = False,
        analysis_only: bool = False
    ):
        """
        Initialize K-1 adapter.

        Args:
            model: The model whose gradients to intercept
            use_hierarchical_tree: If True, use K-1's HierarchicalTree directly
            analysis_only: If True, compute attribution but DON'T scale gradients
                          (Recommended: get interpretability without interfering with Adam)
        """
        self.model = model
        self.use_hierarchical_tree = use_hierarchical_tree
        self.analysis_only = analysis_only

        if use_hierarchical_tree:
            try:
                from k1_system.core import HierarchicalTree
                self._k1_available = True
            except ImportError as e:
                print(f"Warning: Could not import K-1: {e}")
                self._k1_available = False
        else:
            self._k1_available = False

        # Tracking
        self._last_attribution: Dict[str, Any] = {}
        self._update_history: List[Dict[str, Any]] = []

        if analysis_only:
            print("✓ K-1 in ANALYSIS-ONLY mode: Computing attribution without scaling gradients")
    
    def apply_hierarchical_updates(
        self,
        loss_tensor: torch.Tensor,
        current_step: int
    ) -> Dict[str, Any]:
        """
        Apply K-1 style hierarchical error attribution.

        Must be called AFTER loss.backward().

        Args:
            loss_tensor: The loss tensor (gradients already computed)
            current_step: Current training step

        Returns:
            Attribution info dict with:
                - error_path: List of responsible modules
                - nodes_updated: Number of modules updated
                - update_pct: Percentage of model updated
                - analysis_only: Whether gradient scaling was skipped
        """
        if self._k1_available and hasattr(self.model, 'fast_hierarchical_step'):
            # Use K-1's built-in method
            # NOTE: fast_hierarchical_step always scales gradients (can't disable easily)
            # So for analysis_only mode, we just don't call it
            if not self.analysis_only:
                self.model.fast_hierarchical_step(loss_tensor, current_step)
            attribution = self.model.get_error_attribution() if hasattr(self.model, 'get_error_attribution') else {}
        else:
            # Simplified gradient-based attribution
            attribution = self._simple_gradient_attribution(current_step, scale_gradients=not self.analysis_only)

        attribution['analysis_only'] = self.analysis_only

        self._last_attribution = attribution
        self._update_history.append({
            'step': current_step,
            **attribution
        })

        return attribution
    
    def _simple_gradient_attribution(self, current_step: int, scale_gradients: bool = True) -> Dict[str, Any]:
        """
        Simple gradient-proportional attribution (K-1 style).

        Args:
            current_step: Current training step
            scale_gradients: If True, apply proportional scaling (default K-1 behavior)
                           If False, only compute attribution (analysis-only mode)
        """
        # Collect all gradients
        grad_norms = {}
        total_grad = 0.0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                grad_norms[name] = norm
                total_grad += norm

        if total_grad == 0:
            return {
                'error_path': [],
                'nodes_updated': 0,
                'update_pct': 0.0,
                'total_nodes': len(grad_norms)
            }

        # Find top contributors (highest gradient = most responsible)
        sorted_params = sorted(grad_norms.items(), key=lambda x: -x[1])

        # Calculate cumulative contribution
        cumulative = 0.0
        error_path = []
        for name, norm in sorted_params:
            contribution = norm / total_grad
            cumulative += contribution
            error_path.append({
                'name': name,
                'gradient': norm,
                'contribution': contribution
            })
            # Stop when we've accounted for 90% of gradients
            if cumulative >= 0.9:
                break

        # Apply proportional scaling to gradients (ONLY if not analysis-only)
        if scale_gradients:
            self._apply_proportional_scaling(grad_norms, total_grad)

        return {
            'error_path': error_path,
            'nodes_updated': len(error_path),
            'update_pct': len(error_path) / max(1, len(grad_norms)) * 100,
            'total_nodes': len(grad_norms)
        }
    
    def _apply_proportional_scaling(
        self,
        grad_norms: Dict[str, float],
        total_grad: float
    ):
        """
        Scale gradients proportionally (K-1 style sparse updates).
        
        High gradient = full update
        Low gradient = reduced update
        """
        if total_grad == 0:
            return
        
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in grad_norms:
                # Normalize to get proportion
                proportion = grad_norms[name] / total_grad
                
                # Scale: top contributors get 100%, others get proportionally less
                # This creates sparse updates similar to K-1
                scale = min(1.0, proportion * len(grad_norms))
                
                if scale < 1.0:
                    param.grad.mul_(scale)
    
    def get_last_attribution(self) -> Dict[str, Any]:
        """Get attribution from last update."""
        return self._last_attribution
    
    def get_update_statistics(self) -> Dict[str, Any]:
        """Get statistics about updates over time."""
        if not self._update_history:
            return {'total_updates': 0}
        
        avg_nodes = sum(u['nodes_updated'] for u in self._update_history) / len(self._update_history)
        avg_pct = sum(u['update_pct'] for u in self._update_history) / len(self._update_history)
        
        return {
            'total_updates': len(self._update_history),
            'avg_nodes_updated': avg_nodes,
            'avg_update_pct': avg_pct
        }
