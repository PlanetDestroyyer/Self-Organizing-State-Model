"""
Logging utilities for training
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = 'mu_transformer',
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_file: Path to log file (if None, no file logging)
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file provided)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """
    Logger for tracking metrics during training

    Args:
        log_dir: Directory to save metrics
    """

    def __init__(self, log_dir: str = 'results/logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': [],
            'learning_rate': [],
            'step': []
        }

        self.current_epoch = 0

    def log(self, step: int, **kwargs):
        """
        Log metrics at a given step

        Args:
            step: Training step
            **kwargs: Metric name-value pairs
        """
        self.metrics['step'].append(step)

        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def save(self, filename: Optional[str] = None):
        """
        Save metrics to file

        Args:
            filename: Output filename (if None, use timestamp)
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'metrics_{timestamp}.txt'

        filepath = self.log_dir / filename

        with open(filepath, 'w') as f:
            f.write("Step\t" + "\t".join(self.metrics.keys()) + "\n")

            # Get maximum length
            max_len = max(len(v) for v in self.metrics.values() if len(v) > 0)

            for i in range(max_len):
                values = []
                for key in self.metrics.keys():
                    if i < len(self.metrics[key]):
                        values.append(str(self.metrics[key][i]))
                    else:
                        values.append('')
                f.write("\t".join(values) + "\n")

        print(f"Metrics saved to {filepath}")

    def get_latest(self, metric_name: str) -> Optional[float]:
        """
        Get latest value for a metric

        Args:
            metric_name: Name of metric

        Returns:
            Latest value or None if metric doesn't exist
        """
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return self.metrics[metric_name][-1]
        return None


def log_model_info(logger: logging.Logger, model):
    """
    Log model information

    Args:
        logger: Logger instance
        model: Model to log info about
    """
    total_params = model.get_num_params()
    non_emb_params = model.get_num_params(non_embedding=True)

    logger.info(f"Model Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Non-embedding parameters: {non_emb_params:,}")
    logger.info(f"  Model type: {model.__class__.__name__}")
