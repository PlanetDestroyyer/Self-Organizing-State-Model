"""
Loss functions for SOSM training.
"""

from .block_regularization import OrthogonalityLoss, VarianceLoss

__all__ = ['OrthogonalityLoss', 'VarianceLoss']
