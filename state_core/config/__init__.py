"""
Configuration module for state_core.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml (default: config/config.yaml)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Return default config
    return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'stage': 0,
        
        'components': {
            'mu': {
                'enabled': True,
                'vocab_size': 50000,
                'embed_dim': 64,
                'max_seq_len': 512,
                'use_full_model': False,
            },
            'temporal': {
                'enabled': False,
                'time_dim': 32,
                'learning_mode': 'gradient',
            },
            'k1': {
                'enabled': False,
                'use_hierarchical_tree': False,
            },
            'graph': {
                'enabled': False,
                'sequential_edges': True,
                'semantic_edges': False,
                'semantic_threshold': 0.5,
                'random_shortcuts': 0.0,
            },
        },
        
        'model': {
            'hidden_dim': 256,
            'n_layers': 6,
            'n_heads': 4,
            'dropout': 0.1,
        },
        
        'logging': {
            'log_loss': True,
            'log_graph_size': True,
            'log_k1_updates': True,
            'log_interval': 100,
        },
    }


__all__ = ['load_config', 'get_default_config']
