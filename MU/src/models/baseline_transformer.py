"""
Baseline Transformer Model

Standard transformer architecture for comparison with MU Transformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import BaselineConfig


class BaselineTransformerBlock(nn.Module):
    """
    Standard transformer block with multi-head attention and feedforward

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feedforward dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

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

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            mask: Optional attention mask [B, T, T]

        Returns:
            [B, T, d_model]
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=mask,
            need_weights=False
        )
        x = self.norm1(x + attn_out)

        # Feedforward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class BaselineTransformer(nn.Module):
    """
    Standard transformer for language modeling

    Args:
        config: BaselineConfig object
    """

    def __init__(self, config: BaselineConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model) * 0.02)

        # Transformer blocks
        self.layers = nn.ModuleList([
            BaselineTransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])

        # Output layer
        self.output = nn.Sequential(
            nn.LayerNorm(config.d_model, eps=1e-6),
            nn.Linear(config.d_model, config.vocab_size)
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
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
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass

        Args:
            input_ids: Token IDs [B, T]
            attention_mask: Optional attention mask [B, T, T]

        Returns:
            logits: Output logits [B, T, vocab_size]
            None: Placeholder to match MU model interface
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"

        # Token + positional embeddings
        x = self.token_embedding(input_ids)  # [B, T, d_model]
        x = x + self.pos_embedding[:, :T, :]
        x = self.dropout(x)

        # Convert mask format if needed
        # nn.MultiheadAttention expects [T, T] or [B*n_heads, T, T]
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            # Convert from boolean mask to additive mask
            # True -> 0.0 (attend), False -> -inf (don't attend)
            attention_mask = attention_mask.float().masked_fill(
                ~attention_mask, float('-inf')
            ).masked_fill(attention_mask, 0.0)

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, mask=attention_mask)

        # Generate logits
        logits = self.output(x)  # [B, T, vocab_size]

        return logits, None

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


class BaselineTransformerLM(nn.Module):
    """
    Baseline transformer configured for causal language modeling
    """

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.model = BaselineTransformer(config)
        self.config = config

        # Create causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len) * float('-inf'), diagonal=1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass with causal masking

        Args:
            input_ids: Token IDs [B, T]
            attention_mask: Optional padding mask [B, T]

        Returns:
            logits: Output logits [B, T, vocab_size]
            None: Placeholder to match MU model interface
        """
        B, T = input_ids.shape

        # Get causal mask for this sequence length
        causal_mask = self.causal_mask[:T, :T]  # [T, T]

        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask is [B, T], convert to [B, T, T]
            padding_mask = attention_mask.unsqueeze(1).expand(-1, T, -1)  # [B, T, T]
            # Convert boolean to float mask
            padding_mask = padding_mask.float().masked_fill(
                ~padding_mask, float('-inf')
            ).masked_fill(padding_mask, 0.0)
            combined_mask = causal_mask.unsqueeze(0) + padding_mask  # [B, T, T]
        else:
            combined_mask = causal_mask  # [T, T]

        return self.model(input_ids, attention_mask=combined_mask)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively

        Args:
            input_ids: Starting tokens [B, T]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top-k tokens

        Returns:
            Generated token IDs [B, max_length]
        """
        self.model.eval()
        B, T = input_ids.shape

        for _ in range(max_length - T):
            # Get logits for current sequence
            logits, _ = self.forward(input_ids)  # [B, T, vocab_size]

            # Get logits for last position
            logits = logits[:, -1, :] / temperature  # [B, vocab_size]

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if we exceed max sequence length
            if input_ids.size(1) >= self.config.max_seq_len:
                break

        return input_ids

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return number of parameters"""
        return self.model.get_num_params(non_embedding=non_embedding)
