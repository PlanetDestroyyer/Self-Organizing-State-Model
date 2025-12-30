"""
Fair Baseline Transformer for SOSM Comparison

Standard transformer architecture with matched parameter count (~132M).
Designed for rigorous comparison on 3 datasets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TransformerBlock(nn.Module):
    """Standard Transformer block with multi-head attention"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (Pre-LN like SOSM)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            mask: Optional causal mask [T, T]
        Returns:
            [B, T, d_model]
        """
        # Pre-LN + self-attention + residual
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        
        # Pre-LN + feedforward + residual
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out
        
        return x


class BaselineTransformer(nn.Module):
    """
    Standard Transformer for fair comparison with SOSM.
    
    Matched configuration:
    - Total params: ~132M (same as SOSM)
    - 6 layers (same)
    - 1024 hidden dim (same)
    - 8 attention heads (same)
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 280,  # Calculated to exactly match SOSM's 132M params
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Embeddings (match SOSM's factorized approach in total params)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Create causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len) * float('-inf'), diagonal=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights (matching SOSM's init)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [B, T]
            return_state: If True, return dummy state dict for compatibility
            
        Returns:
            logits: Output logits [B, T, vocab_size]
            state: Optional state dict (for SOSM compatibility)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
        
        # Embeddings
        x = self.token_embedding(input_ids)  # [B, T, d_model]
        x = x + self.pos_embedding[:, :T, :]
        x = self.dropout(x)
        
        # Get causal mask
        causal_mask = self.causal_mask[:T, :T]
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        # Return dummy state for compatibility with SOSM interface
        state = None
        if return_state:
            state = {'type': 'baseline', 'reg_losses': {'total_reg': torch.tensor(0.0, device=logits.device)}}
        
        return logits if not return_state else (logits, state)
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Return number of parameters
        
        Args:
            non_embedding: If True, exclude embedding parameters
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.pos_embedding.numel()
        return n_params


if __name__ == "__main__":
    # Test parameter count matching
    model = BaselineTransformer()
    total_params = model.get_num_params()
    non_emb_params = model.get_num_params(non_embedding=True)
    
    print(f"Total params: {total_params:,}")
    print(f"Non-embedding params: {non_emb_params:,}")
    print(f"Target (SOSM): 132,120,241")
    print(f"Difference: {total_params - 132_120_241:,}")
    print(f"Match: {abs(total_params - 132_120_241) / 132_120_241 * 100:.2f}% difference")
    
    # Test forward pass
    batch = torch.randint(0, 50257, (2, 128))
    logits = model(batch)
    print(f"\nForward pass test:")
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: (2, 128, 50257)")
    print(f"✓ Forward pass successful!" if logits.shape == (2, 128, 50257) else "✗ Shape mismatch!")
