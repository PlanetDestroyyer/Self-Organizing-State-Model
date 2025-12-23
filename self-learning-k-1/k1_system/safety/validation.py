"""
Validation system for structural operations.

Validates that structural changes don't hurt performance before committing them.
"""

from typing import Callable, Dict, Optional
import numpy as np


class ValidationSystem:
    """
    Validates structural operations using a holdout validation set.
    """

    def __init__(self,
                 validation_drop_threshold: float = 0.05):
        """
        Initialize validation system.

        Args:
            validation_drop_threshold: Maximum allowed performance drop
        """
        self.validation_drop_threshold = validation_drop_threshold
        self.validation_history = []

    def validate_operation(self,
                          operation_name: str,
                          baseline_performance: float,
                          validation_func: Callable[[], float]) -> tuple:
        """
        Validate a structural operation.

        Args:
            operation_name: Name of operation
            baseline_performance: Performance before operation
            validation_func: Function to compute validation performance

        Returns:
            (is_valid, new_performance, performance_drop) tuple
        """
        # Compute new performance
        new_performance = validation_func()

        # Compute performance drop
        if baseline_performance > 0:
            perf_drop = (baseline_performance - new_performance) / baseline_performance
        else:
            perf_drop = 0.0

        # Validate
        is_valid = perf_drop <= self.validation_drop_threshold

        # Log validation
        self.validation_history.append({
            'operation': operation_name,
            'baseline_performance': baseline_performance,
            'new_performance': new_performance,
            'performance_drop': perf_drop,
            'is_valid': is_valid
        })

        return is_valid, new_performance, perf_drop

    def batch_validate(self,
                      operations: list,
                      baseline_performance: float,
                      validation_func: Callable[[], float]) -> Dict:
        """
        Validate multiple operations.

        Args:
            operations: List of operation names
            baseline_performance: Baseline performance
            validation_func: Validation function

        Returns:
            Dictionary with validation results
        """
        results = {
            'valid_operations': [],
            'invalid_operations': [],
            'performance_changes': []
        }

        for operation in operations:
            is_valid, new_perf, perf_drop = self.validate_operation(
                operation, baseline_performance, validation_func
            )

            if is_valid:
                results['valid_operations'].append(operation)
            else:
                results['invalid_operations'].append(operation)

            results['performance_changes'].append({
                'operation': operation,
                'performance_drop': perf_drop
            })

        return results

    def get_validation_statistics(self) -> Dict:
        """
        Get statistics about validation history.

        Returns:
            Dictionary of statistics
        """
        if not self.validation_history:
            return {
                'total_validations': 0,
                'success_rate': 0.0
            }

        total = len(self.validation_history)
        valid = sum(1 for v in self.validation_history if v['is_valid'])

        perf_drops = [v['performance_drop'] for v in self.validation_history]

        return {
            'total_validations': total,
            'valid_count': valid,
            'invalid_count': total - valid,
            'success_rate': valid / total if total > 0 else 0.0,
            'avg_performance_drop': np.mean(perf_drops),
            'max_performance_drop': np.max(perf_drops),
            'min_performance_drop': np.min(perf_drops)
        }

    def update_threshold(self, new_threshold: float):
        """
        Update validation drop threshold.

        Args:
            new_threshold: New threshold value
        """
        self.validation_drop_threshold = max(0.0, min(1.0, new_threshold))
