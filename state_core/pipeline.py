"""
State Core Pipeline - Main execution pipeline.

Defines exact execution order for forward and backward passes.
Orchestrates all adapters based on current stage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from .state import State
from .stages import StageController
from .adapters import MUAdapter, TemporalAdapter, K1Adapter
from .graph import GraphBuilder, GraphMaskConverter
from .config import load_config, get_default_config


class TransformerLayer(nn.Module):
    """Simple transformer layer for the state core pipeline."""
    
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with optional mask
        normed = self.norm1(x)
        if mask is not None:
            attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        else:
            attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # Feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class StateCorePipeline(nn.Module):
    """
    Main execution pipeline for the Self-Organizing State Model.
    
    Forward pass:
        1. Token IDs → MU Adapter → semantic_state
        2. (Stage 1+) TEMPORAL Adapter → temporal_state
        3. (Stage 3) Graph Builder → routing_state
        4. Attention layers with routing mask
        5. Output logits
    
    Backward pass:
        1. Compute loss
        2. Backprop normally
        3. (Stage 2+) K-1 Adapter intercepts gradients
        4. Apply selective updates
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration dict (or loads from config.yaml)
        """
        super().__init__()
        
        # Load config
        if config is None:
            config = load_config()
        elif isinstance(config, int):
            # If just stage number passed
            config = get_default_config()
            config['stage'] = config
        
        self.config = config
        
        # Stage controller
        stage = config.get('stage', 0)
        self.stage_controller = StageController(stage)
        
        # Get component configs
        mu_cfg = config.get('components', {}).get('mu', {})
        temporal_cfg = config.get('components', {}).get('temporal', {})
        k1_cfg = config.get('components', {}).get('k1', {})
        graph_cfg = config.get('components', {}).get('graph', {})
        model_cfg = config.get('model', {})
        
        # Vocab and dimensions
        self.vocab_size = mu_cfg.get('vocab_size', 50000)
        self.embed_dim = mu_cfg.get('embed_dim', 64)
        self.time_dim = temporal_cfg.get('time_dim', 32)
        self.max_seq_len = mu_cfg.get('max_seq_len', 512)
        
        # Compute model dimension
        self.model_dim = self.embed_dim
        if self.stage_controller.temporal_enabled:
            self.model_dim += self.time_dim
        
        # === ADAPTERS ===
        
        # MU Adapter (always enabled)
        self.mu_adapter = MUAdapter(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            max_seq_len=self.max_seq_len,
            flatten_output=True,
            use_full_model=mu_cfg.get('use_full_model', False)
        )
        
        # TEMPORAL Adapter (Stage 1+)
        self.temporal_adapter = TemporalAdapter(
            vocab_size=self.vocab_size,
            time_dim=self.time_dim,
            learning_mode=temporal_cfg.get('learning_mode', 'gradient')
        )
        
        # Graph Builder (Stage 3)
        self.graph_builder = GraphBuilder(
            enable_sequential=graph_cfg.get('sequential_edges', True),
            enable_semantic=graph_cfg.get('semantic_edges', False),
            enable_shortcuts=graph_cfg.get('random_shortcuts', 0) > 0,
            semantic_threshold=graph_cfg.get('semantic_threshold', 0.5),
            shortcut_prob=graph_cfg.get('random_shortcuts', 0)
        )
        self.graph_mask_converter = GraphMaskConverter()
        
        # === TRANSFORMER LAYERS ===
        hidden_dim = model_cfg.get('hidden_dim', 256)
        n_layers = model_cfg.get('n_layers', 6)
        n_heads = model_cfg.get('n_heads', 4)
        dropout = model_cfg.get('dropout', 0.1)
        
        # Input projection (adapts to varying input dim based on stage)
        self.input_proj = nn.Linear(self.model_dim, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, n_heads, hidden_dim * 4, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.vocab_size)
        
        # K-1 Adapter (created after model is built)
        self.k1_adapter = None
        
        print(f"StateCorePipeline initialized:")
        print(f"  {self.stage_controller}")
        print(f"  Model dim: {self.model_dim}")
        print(f"  Vocab size: {self.vocab_size}")
    
    def _init_k1_adapter(self):
        """Initialize K-1 adapter (called after model is built)."""
        if self.k1_adapter is None:
            self.k1_adapter = K1Adapter(self)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        return_state: bool = True
    ) -> Tuple[torch.Tensor, State]:
        """
        Forward pass through the pipeline.
        
        Args:
            token_ids: [B, T] token indices
            return_state: Whether to return State object
            
        Returns:
            logits: [B, T, vocab_size]
            state: State object (if return_state=True)
        """
        B, T = token_ids.shape
        state = State()
        
        # === Stage 0+: MU Semantic State ===
        state.semantic_state = self.mu_adapter(token_ids)  # [B, T, 64]
        
        # === Stage 1+: TEMPORAL Time State ===
        if self.stage_controller.temporal_enabled:
            state.temporal_state = self.temporal_adapter(
                token_ids,
                semantic_state=None,  # Don't combine yet
                update_time=self.training
            )
            # Combine for model input
            embeddings = torch.cat([state.semantic_state, state.temporal_state], dim=-1)
        else:
            embeddings = state.semantic_state
        
        # === Stage 3: Graph Routing ===
        attention_mask = None
        if self.stage_controller.graph_enabled:
            graph = self.graph_builder.build_graph(
                seq_len=T,
                semantic_state=state.semantic_state,
                batch_size=B
            )
            attention_mask = self.graph_mask_converter.convert_to_additive(
                graph, device=token_ids.device
            )
            state.routing_state = {
                'graph': graph,
                'attention_mask': attention_mask,
                'num_edges': graph['num_edges']
            }
        
        # === Transformer Forward ===
        h = self.input_proj(embeddings)
        
        for layer in self.layers:
            h = layer(h, attention_mask)
        
        h = self.output_norm(h)
        logits = self.output_proj(h)
        
        if return_state:
            return logits, state
        return logits
    
    def backward_with_k1(
        self,
        loss: torch.Tensor,
        state: State,
        current_step: int = 0
    ) -> Dict[str, Any]:
        """
        Backward pass with K-1 gradient interception.
        
        Must be called AFTER loss.backward().
        
        Args:
            loss: Loss tensor (gradients already computed)
            state: State object from forward pass
            current_step: Current training step
            
        Returns:
            Responsibility dict from K-1
        """
        if not self.stage_controller.k1_enabled:
            return {'k1_disabled': True}
        
        # Initialize K-1 adapter if needed
        self._init_k1_adapter()
        
        # Apply hierarchical updates
        responsibility = self.k1_adapter.apply_hierarchical_updates(loss, current_step)
        state.responsibility = responsibility
        
        return responsibility
    
    def set_stage(self, stage: int):
        """
        Change current stage.
        
        Note: May require re-initializing model if dimensions change.
        """
        old_stage = self.stage_controller.stage
        self.stage_controller.set_stage(stage)
        
        # Update model dimension if TEMPORAL toggled
        if (old_stage < 1 <= stage) or (old_stage >= 1 > stage):
            print(f"Warning: Stage change affects model dimension. "
                  f"Re-initialize pipeline for proper dimension handling.")
    
    def get_log_summary(self, state: State, loss: float = None) -> Dict[str, Any]:
        """Get summary for logging."""
        summary = {
            'stage': self.stage_controller.stage,
            **state.log_summary()
        }
        if loss is not None:
            summary['loss'] = loss
        return summary
