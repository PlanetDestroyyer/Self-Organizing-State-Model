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
from .utils.rope import RoPEEmbedding, apply_rotary_pos_emb  # PHASE 2.2: RoPE
from .losses import OrthogonalityLoss, VarianceLoss  # PHASE 2.5: Block regularization
from .layers import PairNorm  # PHASE 2.5: Graph oversmoothing prevention


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
    
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1, use_rope: bool = True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_rope = use_rope
        
        # PHASE 2.2: RoPE for better position encoding
        if use_rope:
            self.rope = RoPEEmbedding(dim=self.head_dim, max_len=8192)
        
        # Pre-norm attention
        self.norm1 = nn.LayerNorm(dim)
        
        # Manual attention (to apply RoPE)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
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
        
        # PHASE 2.5: PairNorm for preventing oversmoothing (set externally)
        self.pair_norm = None
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims (for RoPE)."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Update state representation with RoPE.

        Args:
            x: Projected state [B, T, dim]
            mask: Attention mask from graph routing (None = unrestricted)

        Returns:
            Updated state representation [B, T, dim]
        """
        B, T, D = x.shape
        
        # Pre-norm
        normed = self.norm1(x)
        
        # PHASE 2.2: Manual attention with RoPE
        Q = self.q_proj(normed).view(B, T, self.n_heads, self.head_dim)
        K = self.k_proj(normed).view(B, T, self.n_heads, self.head_dim)
        V = self.v_proj(normed).view(B, T, self.n_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        if self.use_rope:
            cos, sin = self.rope(T)  # [T, head_dim]
            
            # Reshape cos/sin to broadcast correctly with Q/K: [B, T, n_heads, head_dim]
            # cos/sin: [T, head_dim] → [1, T, 1, head_dim]
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, head_dim]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, T, 1, head_dim]
            
            # Apply rotations
            Q = Q * cos + self._rotate_half(Q) * sin
            K = K * cos + self._rotate_half(K) * sin
        
        # Reshape for attention: [B, n_heads, T, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply graph mask if provided
        if mask is not None:
            # mask is [T, T], expand to [B, n_heads, T, T]
            attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_weights, V)  # [B, n_heads, T, head_dim]
        
        # Reshape back: [B, T, dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.out_proj(attn_out)
        
        # Residual
        x = x + attn_out
        
        # PHASE 2.5: PairNorm to prevent oversmoothing
        # Applied after attention to maintain pairwise distances
        if hasattr(self, 'pair_norm') and self.pair_norm is not None:
            x = self.pair_norm(x)
        
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

        # PHASE 2.2: Enable RoPE in operators
        use_rope = model_cfg.get('use_rope', True)  # Default: enabled
        
        # State Update Operators (NOT Transformer layers - computation primitives)
        self.operators = nn.ModuleList([
            StateUpdateOperator(hidden_dim, n_heads, hidden_dim * 4, dropout, use_rope=use_rope)
            for _ in range(n_layers)
        ])
        
        if use_rope:
            print("  ✓ RoPE (Rotary Position Embeddings) enabled")
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.vocab_size)
        
        # PHASE 2.5: Block Regularization to Prevent Semantic Collapse
        # Configuration
        reg_cfg = config.get('regularization', {})
        self.enable_regularization = reg_cfg.get('enabled', False)
        self.lambda_ortho = reg_cfg.get('lambda_ortho', 0.01)
        self.lambda_var = reg_cfg.get('lambda_var', 0.01)
        self.enable_pair_norm = reg_cfg.get('enable_pair_norm', False)
        
        # Loss modules
        if self.enable_regularization:
            self.ortho_loss = OrthogonalityLoss()
            self.var_loss = VarianceLoss(target_std=1.0)
            print(f"  ✓ Block regularization enabled (λ_ortho={self.lambda_ortho}, λ_var={self.lambda_var})")
        else:
            self.ortho_loss = None
            self.var_loss = None
        
        # PairNorm for operators
        if self.enable_pair_norm:
            for operator in self.operators:
                operator.pair_norm = PairNorm(scale=1.0, learnable=False)
            print("  ✓ PairNorm enabled in all operators (prevents oversmoothing)")
        
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
        
        # PHASE 2.5: Compute regularization losses
        reg_losses = {}
        if self.enable_regularization and self.training:
            # Orthogonality loss on semantic state
            ortho_loss_val, ortho_info = self.ortho_loss(state.semantic_state)
            reg_losses['orthogonality'] = ortho_loss_val
            reg_losses['ortho_info'] = ortho_info
            
            # Variance loss on semantic state
            var_loss_val, var_info = self.var_loss(state.semantic_state)
            reg_losses['variance'] = var_loss_val
            reg_losses['var_info'] = var_info
            
            # Weighted total regularization
            reg_losses['total_reg'] = (
                self.lambda_ortho * ortho_loss_val +
                self.lambda_var * var_loss_val
            )
        else:
            reg_losses['total_reg'] = torch.tensor(0.0, device=logits.device)
        
        # Add regularization to state for tracking
        state.reg_losses = reg_losses
        
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
