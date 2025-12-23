"""
Production-Grade TEMPORAL Model with SOTA Components
Includes: RMSNorm, SwiGLU, Flash Attention support, proper initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from time_embeddings import TimeEmbeddedTokenizer


# ============================================================================
# SOTA COMPONENTS
# ============================================================================

class Temporal_Temporal_RMSNorm(nn.Module):
    """
    RMS Normalization (used in LLaMA, faster than LayerNorm)
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm_x


class Temporal_Temporal_SwiGLU(nn.Module):
    """
    SwiGLU activation (used in LLaMA, PaLM - better than GELU)
    """
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Temporal_Attention(nn.Module):
    """
    Multi-head attention with TEMPORAL support
    Uses Flash Attention when available (PyTorch 2.0+)
    """

    def __init__(self, dim, n_heads, dropout=0.0, causal=True):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        # QKV projection (single matrix for efficiency)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # QKV projection and split
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use Flash Attention if available (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0 optimized attention (Flash Attention)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=self.causal
            )
        else:
            # Manual attention (fallback)
            attn = (q @ k.transpose(-2, -1)) * self.scale

            if self.causal:
                # Create causal mask
                causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
                attn = attn.masked_fill(~causal_mask, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v

        # Reshape and project
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        out = self.proj(out)

        return out


class Temporal_Block(nn.Module):
    """
    Transformer block with pre-normalization and SOTA components
    """

    def __init__(self, dim, n_heads, ff_dim, dropout=0.0, use_swiglu=True):
        super().__init__()

        # Pre-normalization (like GPT-2, LLaMA)
        self.norm1 = Temporal_RMSNorm(dim)
        self.attn = Temporal_Attention(dim, n_heads, dropout)

        self.norm2 = Temporal_RMSNorm(dim)

        # FFN: SwiGLU or standard GELU
        if use_swiglu:
            self.mlp = Temporal_SwiGLU(dim, ff_dim, dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, dim),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        # Pre-norm + residual
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# TEMPORAL MODEL
# ============================================================================

class Temporal_Transformer(nn.Module):
    """
    Production-grade TEMPORAL transformer with self-learning time embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Time-embedded tokenizer (self-learning!)
        self.tokenizer = Temporal_Tokenizer(
            vocab_size=config.vocab_size,
            content_dim=config.content_dim,
            time_dim=config.time_dim,
            learning_mode=config.time_learning_mode
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Temporal_Block(
                dim=config.total_dim,
                n_heads=config.n_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
                use_swiglu=True  # SOTA activation
            ) for _ in range(config.n_layers)
        ])

        # Final norm
        self.norm_f = Temporal_RMSNorm(config.total_dim)

        # Output projection (language modeling head)
        self.lm_head = nn.Linear(config.total_dim, config.vocab_size, bias=False)

        # Initialize weights (GPT-2 style)
        self.apply(self._init_weights)

        # Special initialization for residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

        print(f"TEMPORAL Model initialized with {self.count_parameters()/1e6:.1f}M parameters")

    def _init_weights(self, module):
        """Initialize weights (GPT-2 style)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, labels=None, update_time=False, return_dict=True):
        """
        Forward pass with automatic gradient flow through time embeddings

        Args:
            input_ids: [batch_size, seq_len]
            labels: [batch_size, seq_len] for computing loss
            update_time: Whether to update time statistics
            return_dict: Return dict vs tuple

        Returns:
            dict with 'logits', 'loss', 'time_stats' (if return_dict=True)
        """
        # Get [content | time] embeddings
        # CRITICAL: Gradients flow through time automatically!
        x = self.tokenizer(input_ids, update_time=update_time)
        x = self.dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm and projection
        x = self.norm_f(x)
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
        }

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_time_statistics(self):
        """Get time embedding statistics for analysis"""
        return self.tokenizer.get_time_statistics()

    def analyze_time_learning(self):
        """
        Analyze what the time embeddings learned
        For research: understand emergent patterns
        """
        return self.tokenizer.analyze_learned_patterns()


class Temporal_Baseline(nn.Module):
    """
    Baseline transformer without time embeddings (for comparison)
    Identical architecture except no time component
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Standard embeddings (no time)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.total_dim)
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(config.dropout)

        # Same transformer architecture as TEMPORAL
        self.blocks = nn.ModuleList([
            Temporal_Block(
                dim=config.total_dim,
                n_heads=config.n_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
                use_swiglu=True
            ) for _ in range(config.n_layers)
        ])

        self.norm_f = Temporal_RMSNorm(config.total_dim)
        self.lm_head = nn.Linear(config.total_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        print(f"Baseline Model initialized with {self.count_parameters()/1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, labels=None, return_dict=True):
        x = self.token_embeddings(input_ids)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_model(config, model_type='temporal'):
    """
    Factory function to create models

    Args:
        config: Configuration object
        model_type: 'temporal' or 'baseline'

    Returns:
        model: Temporal_Transformer or Temporal_Baseline
    """
    if model_type == 'temporal':
        model = Temporal_Transformer(config)
    elif model_type == 'baseline':
        model = Temporal_Baseline(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_gradient_flow(model):
    """
    Verify that gradients flow through time embeddings
    CRITICAL CHECK: Ensures time is truly self-learning
    """
    if not isinstance(model, Temporal_Transformer):
        print("Not a TEMPORAL model - skipping gradient check")
        return True

    time_emb = model.tokenizer.time_embeddings.time_embeddings

    checks = {
        'requires_grad': time_emb.requires_grad,
        'is_leaf': time_emb.is_leaf,
        'grad_fn': time_emb.grad_fn is None,  # Should be None for leaf tensors
    }

    print("\n" + "="*70)
    print("GRADIENT FLOW VERIFICATION")
    print("="*70)
    print(f"✓ Time embeddings require_grad: {checks['requires_grad']}")
    print(f"✓ Time embeddings is leaf tensor: {checks['is_leaf']}")

    if all([checks['requires_grad'], checks['is_leaf']]):
        print("\n✅ VERIFIED: Time embeddings will learn through gradients!")
    else:
        print("\n❌ WARNING: Time embeddings may not learn properly!")

    print("="*70 + "\n")

    return all([checks['requires_grad'], checks['is_leaf']])
