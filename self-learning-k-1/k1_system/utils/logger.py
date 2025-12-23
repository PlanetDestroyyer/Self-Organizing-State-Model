"""
Logging system for training progress.
"""

import json
from pathlib import Path
from datetime import datetime


class TrainingLogger:
    """
    Logs training progress and events.
    """

    def __init__(self, log_dir: str = 'logs'):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.log"
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.json"

        self.metrics_log = []
        
        # OPTIMIZATION: Buffer logs to reduce file I/O
        self.log_buffer = []
        self.buffer_size = 100  # Flush every 100 entries

    def log(self, message: str, level: str = 'INFO'):
        """
        Log a message (OPTIMIZED with buffering).

        Args:
            message: Message to log
            level: Log level
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"

        # Add to buffer
        self.log_buffer.append(log_entry)
        
        # Flush if buffer is full
        if len(self.log_buffer) >= self.buffer_size:
            self._flush_logs()

        # Also print to console
        print(log_entry.strip())
    
    def _flush_logs(self):
        """Flush buffered logs to disk."""
        if not self.log_buffer:
            return
        
        with open(self.log_file, 'a') as f:
            f.writelines(self.log_buffer)
        self.log_buffer = []

    def log_metrics(self, iteration: int, metrics: dict):
        """
        Log metrics for an iteration.

        Args:
            iteration: Current iteration
            metrics: Metrics dictionary
        """
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }

        self.metrics_log.append(entry)

        # Periodically save to file
        if iteration % 100 == 0:
            self._save_metrics()

    def _save_metrics(self):
        """Save metrics to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)

    def log_phase_transition(self, iteration: int):
        """Log Phase 2 activation."""
        self.log(f"{'=' * 60}", 'INFO')
        self.log(f"🚀 PHASE 2 ACTIVATED at iteration {iteration}", 'INFO')
        self.log(f"Self-Learning Mode Enabled - Parameters Now Adjustable", 'INFO')
        self.log(f"{'=' * 60}", 'INFO')

    def log_structural_operation(self, operation: str, results: dict):
        """
        Log structural operation results.

        Args:
            operation: Operation name
            results: Results dictionary
        """
        self.log(f"Structural Operation: {operation}", 'INFO')
        for key, value in results.items():
            self.log(f"  {key}: {value}", 'INFO')

    def finalize(self):
        """Finalize logging and save all data."""
        self._flush_logs()  # Flush any remaining buffered logs
        self._save_metrics()
        self.log("Training completed - logs saved", 'INFO')
