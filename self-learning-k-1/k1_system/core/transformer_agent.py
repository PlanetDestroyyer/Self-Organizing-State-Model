"""
Transformer-based Agent for the K-1 System.

SOTA architecture with:
- Multi-head self-attention
- Positional encoding
- Layer normalization
- Feed-forward network
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention."""
    
    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Multi-head self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class TransformerAgent(nn.Module):
    """
    SOTA Transformer-based agent for K-1 system.
    
    Features:
    - Multi-head self-attention
    - Positional encoding
    - Layer normalization
    - Residual connections
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, hidden_dim * 2, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(output_dim)
        
        # Trust score (K-1 specific)
        self.trust = 0.3
        
        # Activation tracking
        self.activation_count = 0
        self.child_agents = set()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer agent.
        
        Args:
            x: Input tensor. Can be:
               - (batch_size, seq_len, input_dim) for sequences
               - (batch_size, input_dim) for single vectors
               - (input_dim,) for single vector
        
        Returns:
            Output tensor (batch_size, output_dim) or (output_dim,)
        """
        # Handle different input shapes
        original_shape = x.shape
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(device)
        
        if x.dim() == 1:
            # Single vector -> (1, 1, input_dim)
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_output = True
        elif x.dim() == 2:
            # (batch, input_dim) -> (batch, 1, input_dim)
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Project input
        x = self.input_proj(x)  # (batch, seq, hidden)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Pool over sequence (mean pooling)
        x = x.mean(dim=1)  # (batch, hidden)
        
        # Project to output
        x = self.output_proj(x)  # (batch, output)
        x = self.norm(x)
        
        # Restore original shape if needed
        if squeeze_output:
            x = x.squeeze(0)
        
        return x
    
    def route(self, x: torch.Tensor) -> np.ndarray:
        """Compute routing scores for children.
        
        Args:
            x: Input tensor
        
        Returns:
            Routing scores (one per child)
        """
        n_children = len(self.child_agents)
        if n_children == 0:
            return np.array([])
        
        # Simple routing based on input features
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(device)
        
        # Use first n_children dimensions as routing scores
        if x.dim() > 1:
            x = x.mean(dim=0)
        
        scores = torch.softmax(x[:n_children], dim=0)
        return scores.detach().cpu().numpy()
    
    def record_activation(self, activation: float, iteration: int):
        """Record activation for this agent."""
        self.activation_count += 1
    
    def parameters(self):
        """Return parameters for optimization."""
        return super().parameters()
    
    def named_parameters(self):
        """Return named parameters for optimization."""
        return super().named_parameters()


class TransformerEncoder(nn.Module):
    """
    Full transformer encoder for processing sequences.
    
    Use this as the main encoder in the K-1 system
    instead of mean-pooling embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64
    ):
        super().__init__()
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding (max_seq_len + 1 to accommodate CLS token)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len + 1, dropout)
        
        # Project to hidden dim if different
        if embed_dim != hidden_dim:
            self.input_proj = nn.Linear(embed_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, hidden_dim * 2, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # CLS token for pooling (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Initialize
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Token indices (batch_size, seq_len)
        
        Returns:
            Encoded output (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # Embed tokens
        x = self.embedding(x)  # (batch, seq, embed)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+seq, embed)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Project to hidden dim
        x = self.input_proj(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Use CLS token as output
        x = x[:, 0, :]  # (batch, hidden)
        
        # Project to output
        x = self.output_proj(x)  # (batch, output)
        
        return x


def replace_agents_with_transformer(hierarchy, config):
    """
    Replace standard agents in hierarchy with TransformerAgents.
    
    Args:
        hierarchy: K-1 Hierarchy object
        config: Configuration dict
    
    Returns:
        Updated hierarchy
    """
    from ..core.agent import Agent
    
    input_dim = config['model']['input_dim']
    hidden_dim = config['model']['hidden_dim']
    output_dim = config['model']['output_dim']
    
    def replace_agent(agent):
        """Replace a single agent with TransformerAgent."""
        new_agent = TransformerAgent(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_heads=4,
            n_layers=2
        ).to(device)
        
        # Copy trust and children
        new_agent.trust = agent.trust
        new_agent.child_agents = agent.child_agents
        
        return new_agent
    
    # Replace all agents (recursive)
    def replace_recursive(agent):
        new_agent = replace_agent(agent)
        new_children = set()
        for child in agent.child_agents:
            new_child = replace_recursive(child)
            new_children.add(new_child)
        new_agent.child_agents = new_children
        return new_agent
    
    # Replace root
    hierarchy.root = replace_recursive(hierarchy.root)
    
    return hierarchy
