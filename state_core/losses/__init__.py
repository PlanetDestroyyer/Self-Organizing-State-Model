"""
Loss functions for SOSM training.
"""

from .block_regularization import OrthogonalityLoss, VarianceLoss
from .contrastive_loss import ContextContrastiveLoss, BlockContrastiveLoss
from .auxiliary_tasks import AuxiliaryTaskLoss, BlockUsageBalancing

__all__ = [
    'OrthogonalityLoss', 
    'VarianceLoss', 
    'ContextContrastiveLoss', 
    'BlockContrastiveLoss',
    'AuxiliaryTaskLoss',
    'BlockUsageBalancing'
]
