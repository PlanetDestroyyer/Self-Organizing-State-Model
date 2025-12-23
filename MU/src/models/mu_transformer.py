"""
Meaning Unit (MU) Transformer Model

Complete transformer architecture that operates on MU matrices
instead of dense embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .mu_layer import MUTransformerBlock
from .config import MUConfig


class MUTransformer(nn.Module):
    """
    MU Transformer for language modeling

    This model uses MU matrices [r, c] instead of dense embeddings,
    with specialized attention mechanisms that respect semantic roles.

    Args:
        config: MUConfig object with model hyperparameters
    """

    def __init__(self, config: MUConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.r = config.r
        self.c = config.c
        self.rc = self.r * self.c
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len

        # Token to MU embedding
        # We use a linear layer to map one-hot tokens to MU matrices
        self.token_to_mu = nn.Linear(config.vocab_size, self.rc, bias=False)

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_len, self.r, self.c) * 0.02)

        # Convert sensitivity mask to tensor if needed
        if isinstance(config.sensitivity_mask, list):
            sensitivity_mask = torch.tensor(config.sensitivity_mask, dtype=torch.float32)
        else:
            sensitivity_mask = config.sensitivity_mask

        # MU Transformer blocks
        self.layers = nn.ModuleList([
            MUTransformerBlock(
                r=self.r,
                c=self.c,
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_model // 2,  # Feedforward dimension
                dropout=config.dropout,
                sensitivity_mask=sensitivity_mask
            )
            for _ in range(config.n_layers)
        ])

        # MU to logits
        self.mu_to_logits = nn.Sequential(
            nn.Flatten(start_dim=2),  # [B, T, r*c]
            nn.Linear(self.rc, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model, eps=1e-6),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.vocab_size)
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier/normal initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_mu: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            input_ids: Token IDs [B, T]
            attention_mask: Optional attention mask [B, T]
            return_mu: Whether to return final MU states

        Returns:
            logits: Output logits [B, T, vocab_size]
            MU: Final MU states [B, T, r, c] (if return_mu=True)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"

        # Convert tokens to one-hot and then to MU
        # This is memory efficient for small vocab sizes
        token_onehot = F.one_hot(input_ids, num_classes=self.vocab_size).float()  # [B, T, vocab_size]
        MU = self.token_to_mu(token_onehot)  # [B, T, rc]
        MU = MU.view(B, T, self.r, self.c)  # [B, T, r, c]

        # Add positional encoding
        MU = MU + self.pos_embedding[:, :T, :, :]
        MU = self.dropout(MU)

        # Pass through MU transformer blocks
        for layer in self.layers:
            MU = layer(MU, mask=attention_mask)

        # Convert to logits
        logits = self.mu_to_logits(MU)  # [B, T, vocab_size]

        if return_mu:
            return logits, MU
        else:
            return logits, None

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Return the number of parameters in the model

        Args:
            non_embedding: If True, don't count embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_to_mu.weight.numel()
            n_params -= self.pos_embedding.numel()
        return n_params

    def get_mu_representations(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get MU representations for input tokens (useful for probing)

        Args:
            input_ids: Token IDs [B, T]

        Returns:
            MU representations [B, T, r, c]
        """
        with torch.no_grad():
            _, MU = self.forward(input_ids, return_mu=True)
        return MU

    def analyze_slot_usage(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Analyze variance of each slot position across sequence

        Args:
            input_ids: Token IDs [B, T]

        Returns:
            Variance per slot [r, c]
        """
        MU = self.get_mu_representations(input_ids)  # [B, T, r, c]
        # Compute variance across batch and time dimensions
        variance = MU.var(dim=[0, 1])  # [r, c]
        return variance


class MUTransformerLM(nn.Module):
    """
    MU Transformer configured for causal language modeling

    This adds causal masking to ensure autoregressive generation.
    """

    def __init__(self, config: MUConfig):
        super().__init__()
        self.model = MUTransformer(config)
        self.config = config

        # Create causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).bool()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_mu: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with causal masking

        Args:
            input_ids: Token IDs [B, T]
            attention_mask: Optional padding mask [B, T]
            return_mu: Whether to return final MU states

        Returns:
            logits: Output logits [B, T, vocab_size]
            MU: Final MU states [B, T, r, c] (if return_mu=True)
        """
        B, T = input_ids.shape

        # Get causal mask for this sequence length
        causal_mask = self.causal_mask[:T, :T]  # [T, T]

        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask is [B, T], we need to expand it properly
            # causal_mask is [T, T]
            # We want final mask to be [B, T, T]
            causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)  # [B, T, T]
            padding_mask = attention_mask.unsqueeze(1).expand(-1, T, -1)  # [B, T, T]
            combined_mask = causal_mask & padding_mask
        else:
            combined_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)  # [B, T, T]

        return self.model(input_ids, attention_mask=combined_mask, return_mu=return_mu)

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
            logits, _ = self.forward(input_ids, return_mu=False)  # [B, T, vocab_size]

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
