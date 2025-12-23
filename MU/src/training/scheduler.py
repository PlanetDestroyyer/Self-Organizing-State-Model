"""
Learning rate schedulers
"""
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine learning rate schedule with linear warmup

    Args:
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of base lr
        last_epoch: Last epoch index
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate"""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return [base_lr * lr_scale for base_lr in self.base_lrs]


def get_scheduler(
    optimizer,
    scheduler_type: str = 'cosine_warmup',
    warmup_steps: int = 500,
    total_steps: int = 10000,
    **kwargs
):
    """
    Get learning rate scheduler

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('cosine_warmup', 'constant', etc.)
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        **kwargs: Additional scheduler arguments

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'cosine_warmup':
        return CosineWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=kwargs.get('min_lr_ratio', 0.0)
        )
    elif scheduler_type == 'constant':
        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
