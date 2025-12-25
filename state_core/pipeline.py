"""
State Core Pipeline - Main execution pipeline.

Defines exact execution order for forward and backward passes.
Orchestrates all adapters based on current stage.

DESIGN PRINCIPLES:
- Transformer layers are "State Update Operators" - computation primitives
- Semantic and temporal states remain SEPARATE in State object
- Graph routing uses MU Identity block + positions only
- Position is tracked explicitly, not embedded
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


class StateUpdateOperator(nn.Module):
    """
    Computation operator that updates State under constraints.
    
    This is NOT a Transformer layer in the architectural sense. It uses
    attention mechanics as a computational primitive to update state, but:
    
    WHAT IT DOES:
    - Projects semantic + temporal states into computation space
    - Applies attention under graph routing constraints
    - Updates state representation via gated residuals
    
    WHAT IT DOES NOT DO:
    - Define the model architecture (State does)
    - Learn positional patterns (TEMPORAL does)
    - Modify semantic identity (MU is position-invariant)
    - Access embeddings directly (only projected state)
    """
    
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        
        # Pre-norm attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        
        # Pre-norm feed-forward
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        # NO GATING - use standard residual like proven SOTA models (MU, TEMPORAL)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Update state representation.

        Args:
            x: Projected state [B, T, dim]
            mask: Attention mask from graph routing (None = unrestricted)

        Returns:
            Updated state representation [B, T, dim]
        """
        # Self-attention with optional graph mask
        normed = self.norm1(x)
        if mask is not None:
            attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        else:
            # Unrestricted attention (graph disabled)
            attn_out, _ = self.attn(normed, normed, normed)

        # FIXED: Standard residual (no gating - like MU and TEMPORAL!)
        x = x + attn_out
        
        # Feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class StateProjector(nn.Module):
    """
    Projects separate Semantic and Temporal states into a unified Computation Workspace.

    Supports two combination modes:
    - 'add': Project then add (original design)
    - 'concat': Concatenate then project (like TEMPORAL standalone)
    """

    def __init__(self, semantic_dim: int, temporal_dim: int, compute_dim: int,
                 dropout: float = 0.1, combination_mode: str = 'concat'):
        super().__init__()
        self.combination_mode = combination_mode

        if combination_mode == 'add':
            # Additive combination: project separately then add
            self.proj_semantic = nn.Linear(semantic_dim, compute_dim)
            self.proj_temporal = nn.Linear(temporal_dim, compute_dim)
            self.norm = nn.LayerNorm(compute_dim)
        elif combination_mode == 'concat':
            # TEMPORAL-style: concatenate then project (RECOMMENDED)
            # This preserves both representations like TEMPORAL's proven design
            self.proj_combined = nn.Linear(semantic_dim + temporal_dim, compute_dim)
            # Also need semantic-only projection for Stage 0 (when no temporal)
            self.proj_semantic = nn.Linear(semantic_dim, compute_dim)
            self.norm = nn.LayerNorm(compute_dim)
        else:
            raise ValueError(f"Unknown combination_mode: {combination_mode}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, semantic_state: torch.Tensor, temporal_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Project and combine states.

        Args:
            semantic_state: [B, T, sem_dim]
            temporal_state: [B, T, temp_dim] (Optional)

        Returns:
            computation_workspace: [B, T, compute_dim]
        """
        if temporal_state is None:
            # No temporal state: just project semantic (Stage 0)
            workspace = self.proj_semantic(semantic_state)
        else:
            # Combine semantic + temporal
            if self.combination_mode == 'add':
                # Additive: project independently then add
                workspace = self.proj_semantic(semantic_state) + self.proj_temporal(temporal_state)
            else:  # concat
                # Concatenative: concat then project (TEMPORAL-style, RECOMMENDED)
                combined = torch.cat([semantic_state, temporal_state], dim=-1)
                workspace = self.proj_combined(combined)

        return self.dropout(self.norm(workspace))



class StateCorePipeline(nn.Module):
    """
    Main execution pipeline for the Self-Organizing State Model.
    
    CORRECT EXECUTION FLOW:
    
    1. Input: token_ids + sequence indices
    2. Semantic State (MU): token_id → 8×8 semantic matrix (NO positional encoding)
    3. Temporal State (TEMPORAL): (token_id, position) → temporal embedding
    4. State Assembly: State { semantic_state, temporal_state, position_indices }
    5. Graph Construction: Input = MU Identity block + positions → adjacency + mask
    6. State Update Operators: Project state → compute → update via gated residuals
    7. Output Projection: State → logits
    8. Loss: Cross entropy
    9. Backward: Compute gradients
    10. K-1 Responsibility: Trace gradients hierarchically, scale updates
    11. Parameter Update: optimizer.step()
    
    TOGGLE BEHAVIOR:
    - Graph OFF → unrestricted attention
    - TEMPORAL OFF → no time dependence
    - K-1 OFF → standard backprop
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
            config = get_default_config()
            config['stage'] = config
        
        self.config = config
        
        # Stage controller
        stage = config.get('stage', 0)
        self.stage_controller = StageController(stage)
        
        # Get component configs
        mu_cfg = config.get('components', {}).get('mu', {})
        temporal_cfg = config.get('components', {}).get('temporal', {})
        self.k1_cfg = config.get('components', {}).get('k1', {})  # Store for later use
        graph_cfg = config.get('components', {}).get('graph', {})
        model_cfg = config.get('model', {})
        
        # Vocab and dimensions
        self.vocab_size = mu_cfg.get('vocab_size', 50257)  # GPT-2 vocab
        self.embed_dim = mu_cfg.get('embed_dim', 512)  # INCREASED: was 64, MU tested at 512+
        self.time_dim = temporal_cfg.get('time_dim', 256)  # INCREASED: was 32, TEMPORAL tested at 256+
        self.max_seq_len = mu_cfg.get('max_seq_len', 512)
        
        # Compute model dimension (semantic + temporal if enabled)
        self.model_dim = self.embed_dim
        if self.stage_controller.temporal_enabled:
            self.model_dim += self.time_dim
        
        # === ADAPTERS ===
        
        # MU Adapter (always enabled) - pure semantic, NO positional encoding
        # use_full_model=False starts with simple embeddings (proven to work!)
        self.mu_adapter = MUAdapter(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            max_seq_len=self.max_seq_len,
            flatten_output=True,
            use_full_model=mu_cfg.get('use_full_model', False),  # FIXED: Start simple (False)
            n_layers=mu_cfg.get('mu_layers', 1),  # Reduced to 1 (faster, simpler)
            dropout=model_cfg.get('dropout', 0.1)
        )
        
        # PHASE 2.3: Contextual MU Refinement (optional)
        # Adds local context awareness to position-invariant MU
        # CRITICAL DECISION POINT: Test if context helps (homonym separation >0.05)
        self.use_contextual_refinement = mu_cfg.get('use_contextual_refinement', False)
        if self.use_contextual_refinement:
            from .adapters.contextual_refiner import ContextualMURefinementEfficient
            self.contextual_refiner = ContextualMURefinementEfficient(
                mu_dim=self.embed_dim,
                kernel_size=mu_cfg.get('context_window', 3),
                num_layers=mu_cfg.get('context_layers', 1),
                dropout=model_cfg.get('dropout', 0.1)
            )
            print("  ✓ Contextual MU refinement enabled (3-token window)")
        else:
            self.contextual_refiner = None
        
        # TEMPORAL Adapter (Stage 1+)
        self.temporal_adapter = TemporalAdapter(
            vocab_size=self.vocab_size,
            time_dim=self.time_dim,
            learning_mode=temporal_cfg.get('learning_mode', 'gradient')
        )
        
        # Graph Builder (Stage 3) - uses MU semantic state + positions
        # ENABLED: Semantic edges create connections based on cosine similarity of MU Identity blocks
        # ENABLED: Random shortcuts provide small-world long-range connections
        # This allows "capital" ↔ "India" and "is" ↔ "capital" semantic routing
        self.graph_builder = GraphBuilder(
            enable_sequential=graph_cfg.get('sequential_edges', True),
            enable_semantic=graph_cfg.get('semantic_edges', True),
            enable_shortcuts=graph_cfg.get('random_shortcuts', 0.0) > 0,
            semantic_threshold=graph_cfg.get('semantic_threshold', 0.05),  # FIX: Was 0.3, now 0.05
            semantic_k=graph_cfg.get('semantic_k', 5),  # FIX: Added missing parameter
            semantic_method=graph_cfg.get('semantic_method', 'topk'),  # FIX: Added missing parameter
            shortcut_prob=graph_cfg.get('random_shortcuts', 0.0),  # FIX: Use as probability directly
            use_mutual_knn=graph_cfg.get('use_mutual_knn', True),  # FIX: Added missing parameter
            streaming_topk=graph_cfg.get('streaming_topk', True),  # FIX: Added missing parameter
            semantic_blocks=graph_cfg.get('semantic_blocks', None)  # FIX: Added Phase 2 parameter
        )
        self.graph_mask_converter = GraphMaskConverter()
        
        # === STATE UPDATE OPERATORS ===
        hidden_dim = model_cfg.get('hidden_dim', 1024)  # INCREASED: was 256, now 1024 (match MU/TEMPORAL scale)
        n_layers = model_cfg.get('n_layers', 6)
        n_heads = model_cfg.get('n_heads', 8)  # INCREASED: was 4, now 8 (must divide hidden_dim)
        dropout = model_cfg.get('dropout', 0.1)
        
        # State Projector (Replaces simple Linear input_proj)
        # Supports two modes: 'concat' (TEMPORAL-style, RECOMMENDED) or 'add' (original)
        combination_mode = model_cfg.get('combination_mode', 'concat')  # Default to concat (proven in TEMPORAL)
        self.state_projector = StateProjector(
            semantic_dim=self.embed_dim,
            temporal_dim=self.time_dim,
            compute_dim=hidden_dim,
            dropout=dropout,
            combination_mode=combination_mode
        )

        # State Update Operators (NOT Transformer layers - computation primitives)
        self.operators = nn.ModuleList([
            StateUpdateOperator(hidden_dim, n_heads, hidden_dim * 4, dropout)
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
            # Get K-1 configuration
            analysis_only = self.k1_cfg.get('analysis_only', True)  # Default to True (RECOMMENDED)
            use_hierarchical_tree = self.k1_cfg.get('use_hierarchical_tree', False)

            self.k1_adapter = K1Adapter(
                model=self,  # Pass the pipeline as model
                use_hierarchical_tree=use_hierarchical_tree,
                analysis_only=analysis_only
            )
            
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
        device = token_ids.device
        
        # Initialize State with explicit position tracking
        state = State()
        state.position_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        
        # === Step 1: MU Semantic State (position-invariant) ===
        state.semantic_state = self.mu_adapter(token_ids)  # [B, T, 64]
        
        # PHASE 2.3: Apply contextual refinement if enabled
        if self.use_contextual_refinement and self.contextual_refiner is not None:
            state.semantic_state = self.contextual_refiner(state.semantic_state)
        
        # === Step 2: TEMPORAL Time State (if enabled) ===
        if self.stage_controller.temporal_enabled:
            state.temporal_state = self.temporal_adapter(
                token_ids,
                semantic_state=None,  # No semantic mixing
                update_time=self.training
            )
        
        # === Step 3: Project to Computation Workspace ===
        # CRITICAL: No torch.cat on state components.
        # Project inputs separately, then combine into temporary workspace.
        computation_workspace = self.state_projector(
            state.semantic_state,
            state.temporal_state if self.stage_controller.temporal_enabled else None
        )
        
        # === Step 4: Graph Construction (uses full MU semantic state) ===
        attention_mask = None
        if self.stage_controller.graph_enabled:
            # Get full 64D MU semantic state for semantic similarity
            mu_semantic = state.get_mu_identity_block()  # [B, T, 64] - full semantic state
            
            graph = self.graph_builder.build_graph(
                seq_len=T,
                semantic_state=mu_semantic,  # Full 64D semantic state for cosine similarity
                batch_size=B
            )
            attention_mask = self.graph_mask_converter.convert_to_additive(
                graph, device=device
            )
            state.routing_state = {
                'graph': graph,
                'attention_mask': attention_mask,
                'num_edges': graph['num_edges']
            }
        
        # === Step 5: State Update Operators ===
        # Use computation workspace as initial hidden state
        h = computation_workspace
        
        for operator in self.operators:
            h = operator(h, attention_mask)
        
        # === Step 6: Output Projection ===
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
        
        TOGGLE BEHAVIOR:
        - Stage 0: MU only
        - Stage 1: MU + TEMPORAL (adds time dependence)
        - Stage 2: MU + TEMPORAL + K-1 (hierarchical attribution)
        - Stage 3: Full system with graph routing
        """
        old_stage = self.stage_controller.stage
        self.stage_controller.set_stage(stage)
        
        # Warn if dimension changes
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
