"""
Model checkpointing utilities
"""
import torch
from pathlib import Path
from typing import Dict, Optional
import logging


class CheckpointManager:
    """
    Manages model checkpoints during training

    Args:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
    """

    def __init__(
        self,
        checkpoint_dir: str = 'results/checkpoints',
        max_checkpoints: int = 5
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

        self.logger = logging.getLogger('checkpoint_manager')

    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        step: int,
        metrics: Dict,
        config: Dict,
        is_best: bool = False,
        filename: Optional[str] = None
    ):
        """
        Save model checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            step: Current training step
            metrics: Dictionary of metrics
            config: Model configuration
            is_best: Whether this is the best model so far
            filename: Custom filename (if None, use default naming)
        """
        if filename is None:
            filename = f'checkpoint_epoch{epoch}_step{step}.pt'

        filepath = self.checkpoint_dir / filename

        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'config': config
        }

        # Save checkpoint
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")

        # Track checkpoint
        self.checkpoints.append(filepath)

        # Remove old checkpoints if exceeded max
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")

    def load_checkpoint(
        self,
        filepath: str,
        model,
        optimizer=None,
        scheduler=None,
        device: str = 'cpu'
    ) -> Dict:
        """
        Load checkpoint

        Args:
            filepath: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load checkpoint to

        Returns:
            Dictionary with checkpoint information
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Model state loaded from {filepath}")

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("Optimizer state loaded")

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("Scheduler state loaded")

        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {})
        }

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get path to latest checkpoint

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        if self.checkpoints:
            return self.checkpoints[-1]
        return None

    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get path to best checkpoint

        Returns:
            Path to best checkpoint or None if it doesn't exist
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            return best_path
        return None


def save_model_only(
    model,
    filepath: str,
    config: Optional[Dict] = None
):
    """
    Save only model weights (for final export)

    Args:
        model: Model to save
        filepath: Output filepath
        config: Optional configuration dictionary
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config
    }

    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model_only(
    model,
    filepath: str,
    device: str = 'cpu'
) -> Dict:
    """
    Load only model weights

    Args:
        model: Model to load weights into
        filepath: Checkpoint filepath
        device: Device to load to

    Returns:
        Configuration dictionary if present
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return checkpoint.get('config', {})
