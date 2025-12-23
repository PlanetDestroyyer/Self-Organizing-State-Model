"""
Metrics computation and tracking.
"""

from typing import List, Dict
import numpy as np
from collections import deque


class MetricsTracker:
    """
    Tracks and computes system metrics.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics tracker.

        Args:
            window_size: Size of rolling window for metrics
        """
        self.window_size = window_size

        self.accuracy_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.perplexity_history = deque(maxlen=window_size)
        self.trust_history = deque(maxlen=window_size)
        self.agent_count_history = deque(maxlen=window_size)

    def compute_perplexity(self, loss: float) -> float:
        """
        Compute perplexity from loss.

        Perplexity = exp(loss). Clip to prevent overflow.

        Args:
            loss: Cross-entropy loss

        Returns:
            Perplexity value
        """
        # Clip loss to prevent overflow
        clipped_loss = min(loss, 100.0)
        return np.exp(clipped_loss)

    def update(self,
              iteration: int,
              accuracy: float,
              loss: float,
              avg_trust: float,
              total_agents: int):
        """
        Update metrics.

        Args:
            iteration: Current iteration
            accuracy: Current accuracy
            loss: Current loss
            avg_trust: Average trust
            total_agents: Total number of agents
        """
        self.accuracy_history.append(accuracy)
        self.loss_history.append(loss)

        # Compute and track perplexity
        perplexity = self.compute_perplexity(loss)
        self.perplexity_history.append(perplexity)

        self.trust_history.append(avg_trust)
        self.agent_count_history.append(total_agents)

    def get_current_metrics(self) -> Dict:
        """
        Get current metric values.

        Returns:
            Dictionary of current metrics
        """
        if not self.accuracy_history:
            return {}

        return {
            'accuracy': self.accuracy_history[-1],
            'loss': self.loss_history[-1],
            'perplexity': self.perplexity_history[-1] if self.perplexity_history else 0.0,
            'avg_trust': self.trust_history[-1],
            'total_agents': self.agent_count_history[-1]
        }

    def get_rolling_stats(self) -> Dict:
        """
        Get rolling statistics.

        Returns:
            Dictionary of rolling statistics
        """
        if not self.accuracy_history:
            return {}

        return {
            'avg_accuracy': np.mean(self.accuracy_history),
            'std_accuracy': np.std(self.accuracy_history),
            'avg_loss': np.mean(self.loss_history),
            'std_loss': np.std(self.loss_history),
            'avg_perplexity': np.mean(self.perplexity_history) if self.perplexity_history else 0.0,
            'std_perplexity': np.std(self.perplexity_history) if self.perplexity_history else 0.0,
            'avg_trust': np.mean(self.trust_history),
            'avg_agents': np.mean(self.agent_count_history)
        }

    def get_improvement_rate(self, window: int = 100) -> float:
        """
        Get rate of improvement in recent window.

        Args:
            window: Window size

        Returns:
            Improvement rate
        """
        if len(self.accuracy_history) < window:
            return 0.0

        recent = list(self.accuracy_history)[-window:]
        return (recent[-1] - recent[0]) / window

    def compute_performance_summary(self) -> Dict:
        """
        Compute comprehensive performance summary.

        Returns:
            Dictionary with performance summary
        """
        if not self.accuracy_history:
            return {}

        accuracy_list = list(self.accuracy_history)
        loss_list = list(self.loss_history)

        return {
            'best_accuracy': max(accuracy_list),
            'current_accuracy': accuracy_list[-1],
            'worst_accuracy': min(accuracy_list),
            'best_loss': min(loss_list),
            'current_loss': loss_list[-1],
            'worst_loss': max(loss_list),
            'accuracy_improvement': accuracy_list[-1] - accuracy_list[0] if len(accuracy_list) > 1 else 0,
            'total_samples': len(accuracy_list)
        }
