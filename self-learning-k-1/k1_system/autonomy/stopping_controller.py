"""
Autonomous stopping controller.

Determines when the system should autonomously stop training based on
performance plateaus, goal achievement, or diminishing returns.
"""

from typing import List, Optional, Tuple
import numpy as np
from collections import deque


class StoppingController:
    """
    Manages autonomous stopping decisions.
    """

    def __init__(self,
                 max_iterations: int = 500000,
                 patience: int = 50000,
                 min_improvement: float = 0.001,
                 goal_accuracy: Optional[float] = None):
        """
        Initialize stopping controller.

        Args:
            max_iterations: Maximum iterations before forced stop
            patience: Iterations without improvement before stopping
            min_improvement: Minimum improvement to reset patience
            goal_accuracy: Target accuracy (stop when reached)
        """
        self.max_iterations = max_iterations
        self.patience = patience
        self.min_improvement = min_improvement
        self.goal_accuracy = goal_accuracy

        self.best_accuracy = 0.0
        self.best_iteration = 0
        self.iterations_since_improvement = 0
        self.accuracy_history = deque(maxlen=10000)
        self.loss_history = deque(maxlen=10000)

        self.stopped = False
        self.stopping_reason = None

    def update(self, iteration: int, accuracy: float, loss: float) -> Tuple[bool, Optional[str]]:
        """
        Update stopping controller and check if should stop.

        Args:
            iteration: Current iteration
            accuracy: Current accuracy
            loss: Current loss

        Returns:
            (should_stop, reason) tuple
        """
        # Record metrics
        self.accuracy_history.append(accuracy)
        self.loss_history.append(loss)

        # Check for improvement
        if accuracy > self.best_accuracy + self.min_improvement:
            self.best_accuracy = accuracy
            self.best_iteration = iteration
            self.iterations_since_improvement = 0
        else:
            self.iterations_since_improvement = iteration - self.best_iteration

        # Check stopping criteria
        should_stop, reason = self._check_stopping_criteria(iteration, accuracy, loss)

        if should_stop:
            self.stopped = True
            self.stopping_reason = reason

        return should_stop, reason

    def _check_stopping_criteria(self,
                                 iteration: int,
                                 accuracy: float,
                                 loss: float) -> Tuple[bool, Optional[str]]:
        """
        Check all stopping criteria.

        Args:
            iteration: Current iteration
            accuracy: Current accuracy
            loss: Current loss

        Returns:
            (should_stop, reason) tuple
        """
        # Criterion 1: Max iterations reached
        if iteration >= self.max_iterations:
            return True, 'max_iterations_reached'

        # Criterion 2: Goal achieved
        if self.goal_accuracy and accuracy >= self.goal_accuracy:
            return True, 'goal_achieved'

        # Criterion 3: Patience exhausted (plateau)
        if self.iterations_since_improvement >= self.patience:
            return True, 'plateau'

        # Criterion 4: Diminishing returns
        if self._check_diminishing_returns():
            return True, 'diminishing_returns'

        # Criterion 5: Converged (loss very stable)
        if self._check_convergence():
            return True, 'converged'

        return False, None

    def _check_diminishing_returns(self) -> bool:
        """
        Check if improvements are becoming negligible.

        Returns:
            True if diminishing returns detected
        """
        if len(self.accuracy_history) < 5000:
            return False

        # Compare recent improvement rate to earlier improvement rate
        recent = list(self.accuracy_history)[-1000:]
        earlier = list(self.accuracy_history)[-5000:-4000]

        recent_improvement = np.mean(recent) - np.mean(earlier)

        # If improvement is very small over 4000 iterations
        if 0 < recent_improvement < 0.001:
            return True

        return False

    def _check_convergence(self) -> bool:
        """
        Check if loss has converged (very stable).

        Returns:
            True if converged
        """
        if len(self.loss_history) < 1000:
            return False

        recent_loss = list(self.loss_history)[-1000:]
        loss_std = np.std(recent_loss)

        # If loss variance is very low and we've trained for a while
        if loss_std < 0.001 and len(self.loss_history) >= 5000:
            return True

        return False

    def reset_patience(self):
        """Reset patience counter (called on significant improvement)."""
        self.iterations_since_improvement = 0

    def set_goal_accuracy(self, goal: float):
        """
        Set target accuracy goal.

        Args:
            goal: Target accuracy (0.0 to 1.0)
        """
        self.goal_accuracy = goal

    def get_status(self) -> dict:
        """
        Get current stopping controller status.

        Returns:
            Dictionary with status information
        """
        return {
            'stopped': self.stopped,
            'stopping_reason': self.stopping_reason,
            'best_accuracy': self.best_accuracy,
            'best_iteration': self.best_iteration,
            'iterations_since_improvement': self.iterations_since_improvement,
            'patience_remaining': self.patience - self.iterations_since_improvement
        }

    def should_continue(self, iteration: int, accuracy: float, loss: float) -> bool:
        """
        Simplified check: should training continue?

        Args:
            iteration: Current iteration
            accuracy: Current accuracy
            loss: Current loss

        Returns:
            True if should continue
        """
        should_stop, _ = self.update(iteration, accuracy, loss)
        return not should_stop
