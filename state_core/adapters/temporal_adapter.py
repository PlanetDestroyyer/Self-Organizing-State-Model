"""
TEMPORAL Adapter - Wraps TEMPORAL time embeddings.

Imports TEMPORAL exactly as implemented, injects time embeddings.
Ensures gradients flow through time embeddings.
Does NOT alter TEMPORAL internals.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional

# Add TEMPORAL repo to path - use absolute resolved path
TEMPORAL_PATH = (Path(__file__).resolve().parent.parent.parent / "TEMPORAL").resolve()
if str(TEMPORAL_PATH) not in sys.path:
    sys.path.insert(0, str(TEMPORAL_PATH))


class TemporalAdapter(nn.Module):
    """
    Adapter for TEMPORAL time embeddings.
    
    Wraps TEMPORAL's time embedding layer, augments semantic state.
    Ensures time embeddings have requires_grad=True.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        time_dim: int = 32,
        learning_mode: str = 'gradient'
    ):
        """
        Initialize TEMPORAL adapter.
        
        Args:
            vocab_size: Vocabulary size
            time_dim: Time embedding dimension
            learning_mode: 'gradient' or 'hybrid'
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.time_dim = time_dim
        
        # Try to import TEMPORAL's time embeddings
        try:
            # Import from TEMPORAL module using importlib for reliable loading
            import importlib.util
            from pathlib import Path
            
            # Robust path search algorithm
            current_file = Path(__file__).resolve()
            search_start = current_file.parent
            
            possible_files = []
            
            # 1. Walk up to find repo root or TEMPORAL dir
            temp_path = search_start
            for _ in range(5):  # Go up 5 levels max
                # Check for TEMPORAL directory
                if (temp_path / "TEMPORAL" / "time_embeddings.py").exists():
                    possible_files.append(temp_path / "TEMPORAL" / "time_embeddings.py")
                # Check if we are inside TEMPORAL already
                if temp_path.name == "TEMPORAL" and (temp_path / "time_embeddings.py").exists():
                    possible_files.append(temp_path / "time_embeddings.py")
                temp_path = temp_path.parent
                
            # 2. Add standard/environment-specific paths
            env_paths = [
                Path("/kaggle/working/Self-Organizing-State-Model/TEMPORAL/time_embeddings.py"),
                Path("/content/Self-Organizing-State-Model/TEMPORAL/time_embeddings.py"),  # Colab
                Path("TEMPORAL/time_embeddings.py").resolve(),
            ]
            possible_files.extend([p for p in env_paths if p.exists()])
            
            time_emb_file = possible_files[0] if possible_files else None
            
            if time_emb_file and time_emb_file.exists():
                spec = importlib.util.spec_from_file_location("time_embeddings", str(time_emb_file))
                time_embeddings_module = importlib.util.module_from_spec(spec)
                sys.modules["time_embeddings"] = time_embeddings_module
                spec.loader.exec_module(time_embeddings_module)
                
                Temporal_TimeEmbeddings = time_embeddings_module.Temporal_TimeEmbeddings
                self.time_embeddings = Temporal_TimeEmbeddings(
                    vocab_size=vocab_size,
                    time_dim=time_dim,
                    learning_mode=learning_mode
                )
                self._using_temporal = True
                print(f"✓ TEMPORAL time embeddings loaded successfully from {time_emb_file}")
            else:
                # Debug info
                cw = Path.cwd()
                raise FileNotFoundError(f"time_embeddings.py not found. CWD: {cw}. Checked up to 5 levels up and standard env paths.")
        except Exception as e:
            print(f"Note: Using simple time embeddings (TEMPORAL import: {type(e).__name__}: {e})")
            self._using_temporal = False
            self._init_simple_time_embeddings(vocab_size, time_dim)
    
    def _init_simple_time_embeddings(self, vocab_size: int, time_dim: int):
        """Initialize simple time embeddings (TEMPORAL-style)."""
        # Time embeddings - initialized to zeros, learns from scratch
        self.time_embeddings_param = nn.Parameter(
            torch.zeros(vocab_size, time_dim),
            requires_grad=True  # Gradients flow through
        )
    
    def forward(
        self,
        token_ids: torch.Tensor,
        semantic_state: Optional[torch.Tensor] = None,
        update_time: bool = False
    ) -> torch.Tensor:
        """
        Get time embeddings and optionally combine with semantic state.
        
        Args:
            token_ids: [B, T] token indices
            semantic_state: [B, T, semantic_dim] optional semantic state
            update_time: Whether to update time statistics
            
        Returns:
            temporal_state: [B, T, time_dim] or combined [B, T, semantic_dim + time_dim]
        """
        if self._using_temporal:
            time_emb = self.time_embeddings(
                token_ids,
                update_time=update_time
            )
        else:
            time_emb = self.time_embeddings_param[token_ids]
        
        # Combine with semantic state if provided
        if semantic_state is not None:
            return torch.cat([semantic_state, time_emb], dim=-1)
        
        return time_emb
    
    def get_time_dim(self) -> int:
        """Get time embedding dimension."""
        return self.time_dim
    
    def verify_gradient_flow(self) -> bool:
        """Verify that time embeddings have gradients enabled."""
        if self._using_temporal:
            param = self.time_embeddings.time_embeddings
        else:
            param = self.time_embeddings_param
        
        return param.requires_grad and param.is_leaf
