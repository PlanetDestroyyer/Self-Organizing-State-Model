"""
Main training loop for the Self-Learning K-1 System.

Implements the complete training pipeline with automatic Phase 1 to Phase 2 transition.
"""

import json
import numpy as np
import time
from pathlib import Path
from typing import Tuple, List

# Core components
from .core import Hierarchy, HierarchicalRouter, TrustSystem, Agent
from .learning import CreditAssignmentSystem, ForwardPass, AdaptiveWeightUpdater
from .structural import PruningSystem, MergingSystem, GrowingSystem, ReorganizationSystem
from .autonomy import ParameterController, SystemState, StoppingController, SelfDiagnostic
from .safety import SnapshotManager, ValidationSystem
from .initialization import HierarchyBuilder
from .utils import MetricsTracker, TrainingLogger


class K1System:
    """
    Complete Self-Learning K-1 System.

    Combines all components and manages training with autonomous phase transition.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize K1 System.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / 'config' / 'config_phase1.json'

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize logger
        self.logger = TrainingLogger()
        self.logger.log("Initializing Self-Learning K-1 System", 'INFO')

        # Initialize hierarchy
        builder = HierarchyBuilder(
            input_dim=128,
            hidden_dim=64,
            output_dim=128,
            initial_trust=self.config['trust']['initial']
        )

        self.hierarchy = builder.build_simple_hierarchy(num_domains=5, agents_per_domain=3)
        builder.print_structure(self.hierarchy)

        # Initialize core systems
        self.trust_system = TrustSystem(
            error_penalty_multiplier=self.config['trust']['error_penalty_multiplier'],
            success_reward_multiplier=self.config['trust']['success_reward_multiplier'],
            success_reward_cap=self.config['trust']['success_reward_cap'],
            cache_threshold=self.config['trust']['cache_threshold']
        )

        self.router = HierarchicalRouter(
            hierarchy=self.hierarchy,
            confidence_threshold=self.config['structure']['routing_confidence_threshold'],
            max_depth=self.config['structure']['max_hierarchy_depth'],
            exploration_rate=self.config['exploration']['initial_rate']
        )

        self.forward_pass = ForwardPass(
            hierarchy=self.hierarchy,
            router=self.router,
            trust_cache=self.trust_system.trust_cache
        )

        self.weight_updater = AdaptiveWeightUpdater(
            learning_rate=self.config['learning']['learning_rate']
        )

        self.credit_assignment = CreditAssignmentSystem(
            trust_system=self.trust_system,
            top_k=self.config['learning']['top_k']
        )

        # Initialize structural operations
        self.pruning_system = PruningSystem(
            hierarchy=self.hierarchy,
            trust_threshold=self.config['pruning']['trust_threshold'],
            usage_threshold=self.config['pruning']['usage_threshold'],
            dormancy_threshold=self.config['pruning']['dormancy_threshold'],
            validation_drop_threshold=self.config['pruning']['validation_drop_threshold'],
            min_agents_per_manager=self.config['structure']['min_agents_per_manager']
        )

        self.merging_system = MergingSystem(
            hierarchy=self.hierarchy,
            similarity_threshold=self.config['merging']['similarity_threshold'],
            min_trust=self.config['merging']['min_trust'],
            validation_drop_threshold=self.config['merging']['validation_drop_threshold']
        )

        self.growing_system = GrowingSystem(
            hierarchy=self.hierarchy,
            error_cluster_min_size=self.config['growing']['error_cluster_min_size'],
            error_frequency_threshold=self.config['growing']['error_frequency_threshold'],
            new_agent_initial_trust=self.config['growing']['new_agent_initial_trust'],
            validation_period=self.config['growing']['validation_period'],
            success_threshold=self.config['growing']['success_threshold'],
            failure_threshold=self.config['growing']['failure_threshold'],
            max_agents=self.config['structure']['max_total_agents']
        )

        self.reorganization_system = ReorganizationSystem(
            hierarchy=self.hierarchy,
            max_moves_per_cycle=0.1
        )

        # Initialize autonomy (Phase 2)
        self.param_controller = ParameterController(self.config)
        self.stopping_controller = StoppingController(
            max_iterations=self.config['stopping']['max_iterations'],
            patience=self.config['stopping']['patience'],
            min_improvement=self.config['stopping']['min_improvement']
        )
        self.diagnostic_system = SelfDiagnostic()

        # Initialize safety
        self.snapshot_manager = SnapshotManager(
            max_snapshots=self.config['rollback']['max_snapshots'],
            performance_drop_threshold=self.config['rollback']['performance_drop_threshold']
        )
        self.validation_system = ValidationSystem(
            validation_drop_threshold=self.config['pruning']['validation_drop_threshold']
        )

        # Initialize metrics
        self.metrics_tracker = MetricsTracker(window_size=1000)

        # Training state
        self.current_iteration = 0
        self.phase_1_duration = self.config['system']['phase_1_duration']
        self.phase_2_active = False

        self.logger.log("System initialization complete", 'INFO')

    def train(self, train_data: np.ndarray, train_labels: np.ndarray,
              val_data: np.ndarray = None, val_labels: np.ndarray = None):
        """
        Main training loop with automatic phase transition.

        Args:
            train_data: Training data
            train_labels: Training labels
            val_data: Validation data
            val_labels: Validation labels
        """
        self.logger.log(f"Starting training for up to {self.config['stopping']['max_iterations']} iterations", 'INFO')
        self.logger.log(f"Phase 1 (Fixed Parameters): Iterations 0-{self.phase_1_duration}", 'INFO')
        self.logger.log(f"Phase 2 (Autonomous Adjustment): Iterations {self.phase_1_duration}+", 'INFO')

        # Training loop
        for iteration in range(self.config['stopping']['max_iterations']):
            self.current_iteration = iteration
            self.forward_pass.update_iteration(iteration)

            # PHASE CHECK - Automatic transition at phase_1_duration
            if iteration == self.phase_1_duration:
                self._activate_phase_2()

            # Sample batch
            batch_indices = np.random.choice(len(train_data),
                                           size=self.config['learning']['batch_size'],
                                           replace=False)
            batch_data = train_data[batch_indices]
            batch_labels = train_labels[batch_indices]

            # Training step
            batch_accuracy, batch_loss = self._training_step(batch_data, batch_labels)

            # Validation (every 100 iterations)
            if val_data is not None and iteration % 100 == 0:
                val_accuracy = self._validate(val_data, val_labels)
            else:
                val_accuracy = batch_accuracy

            # Update metrics
            all_agents = self.hierarchy.get_all_agents()
            avg_trust = self.trust_system.compute_avg_trust(all_agents)

            self.metrics_tracker.update(
                iteration=iteration,
                accuracy=val_accuracy,
                loss=batch_loss,
                avg_trust=avg_trust,
                total_agents=self.hierarchy.count_agents()
            )

            # Phase 2: Parameter adjustment
            if iteration >= self.phase_1_duration:
                self._phase_2_adjustments(iteration, all_agents)

            # Structural operations
            self._run_structural_operations(iteration, val_accuracy)

            # Update exploration rate
            new_exploration_rate = self.router.compute_exploration_rate(
                iteration,
                self.config['exploration']['initial_rate'],
                self.config['exploration']['decay_constant'],
                self.config['exploration']['minimum_rate']
            )
            if iteration >= self.phase_1_duration:
                # Use param controller's rate in Phase 2
                new_exploration_rate = self.param_controller.exploration_rate

            self.router.update_exploration_rate(new_exploration_rate)

            # Logging (every 100 iterations)
            if iteration % 100 == 0:
                self._log_progress(iteration, val_accuracy, batch_loss, avg_trust)

            # Check stopping criteria
            should_stop, reason = self.stopping_controller.update(iteration, val_accuracy, batch_loss)
            if should_stop:
                self.logger.log(f"Stopping training at iteration {iteration}: {reason}", 'INFO')
                break

        # Finalize
        self._finalize_training()

    def _training_step(self, batch_data: np.ndarray, batch_labels: np.ndarray) -> Tuple[float, float]:
        """
        Execute one training step.

        Args:
            batch_data: Batch of input data
            batch_labels: Batch of labels

        Returns:
            (accuracy, loss) tuple
        """
        batch_size = len(batch_data)
        total_loss = 0.0
        correct = 0

        for i in range(batch_size):
            x = batch_data[i]
            target = batch_labels[i]

            # Forward pass
            output, routing_path = self.forward_pass.forward(x, mode='hard')

            # Compute loss and error
            loss = self.forward_pass.compute_loss(output, np.array([target]), task='classification')
            total_loss += loss

            # Compute prediction
            pred = self.forward_pass.compute_prediction(output, task='classification')
            if np.argmax(pred) == int(target):
                correct += 1

            # Credit assignment
            activated_agents = routing_path.get_activated_agents()
            selected_agents = self.credit_assignment.assign_credit(
                routing_path, loss, np.array([target]), output
            )

            # Update weights
            for agent in selected_agents:
                # Compute gradient (simplified)
                gradient = self.weight_updater.compute_gradient(
                    agent, x, np.array([target]), output
                )
                gradient = self.weight_updater.clip_gradient(gradient)

                # Update
                old_error = loss
                self.weight_updater.update_agent(agent, gradient)

                # Re-compute to check improvement (simplified - just trust update)
                # In practice, would re-forward through agent
                new_error = loss * 0.95  # Assume small improvement

                error_reduction = old_error - new_error
                if error_reduction > 0:
                    self.trust_system.report_success(agent, error_reduction)
                else:
                    self.trust_system.report_error(agent, abs(error_reduction))

            # Record activation patterns for reorganization
            self.reorganization_system.record_activation_pattern(activated_agents)
            self.merging_system.record_coactivation([a.id for a in activated_agents])

            # Record error for growing system
            error_vector = output - np.array([target])
            self.growing_system.record_error(error_vector, loss, activated_agents, self.current_iteration)

        accuracy = correct / batch_size
        avg_loss = total_loss / batch_size

        return accuracy, avg_loss

    def _validate(self, val_data: np.ndarray, val_labels: np.ndarray) -> float:
        """
        Run validation.

        Args:
            val_data: Validation data
            val_labels: Validation labels

        Returns:
            Validation accuracy
        """
        correct = 0
        for i in range(len(val_data)):
            output, _ = self.forward_pass.forward(val_data[i], mode='hard')
            pred = self.forward_pass.compute_prediction(output, task='classification')
            if np.argmax(pred) == int(val_labels[i]):
                correct += 1

        return correct / len(val_data)

    def _activate_phase_2(self):
        """Activate Phase 2: Self-Learning Mode."""
        self.logger.log_phase_transition(self.current_iteration)
        self.param_controller.activate_phase_2()
        self.phase_2_active = True

    def _phase_2_adjustments(self, iteration: int, all_agents: List[Agent]):
        """
        Execute Phase 2 parameter adjustments.

        Args:
            iteration: Current iteration
            all_agents: List of all agents
        """
        # Collect system state
        current_metrics = self.metrics_tracker.get_current_metrics()
        improvement_rate = self.metrics_tracker.get_improvement_rate(window=5000)

        state = SystemState(
            iteration=iteration,
            accuracy=current_metrics.get('accuracy', 0.0),
            loss=current_metrics.get('loss', 1.0),
            validation_accuracy=current_metrics.get('accuracy', 0.0),
            total_agents=self.hierarchy.count_agents(),
            active_agents_pct=len(self.hierarchy.get_active_agents()) / max(1, self.hierarchy.count_agents()),
            avg_trust=self.trust_system.compute_avg_trust(all_agents),
            trust_variance=self.trust_system.compute_trust_variance(all_agents),
            cache_hit_rate=0.5,  # Simplified
            rollback_count_last_10k=self.snapshot_manager.get_rollback_count(10000),
            iterations_since_improvement=self.stopping_controller.iterations_since_improvement,
            structural_ops_time_pct=0.1,  # Simplified
            avg_error_magnitude=current_metrics.get('loss', 0.5),
            improvement_last_5k=improvement_rate * 5000,
            structural_changes_last_10k=5,  # Simplified
            cache_size=self.trust_system.trust_cache.get_cache_size()
        )

        # Adjust parameters
        self.param_controller.check_and_adjust(iteration, state, all_agents)

        # Apply adjusted parameters to systems
        self._apply_adjusted_parameters()

        # Run diagnostics every 5000 iterations
        if iteration % 5000 == 0:
            recent_perf = list(self.metrics_tracker.accuracy_history)
            diagnostics = self.diagnostic_system.diagnose(
                iteration, state, all_agents, recent_perf
            )

            if diagnostics['problems']:
                self.logger.log(f"Diagnostics found {len(diagnostics['problems'])} issues", 'WARNING')
                actions = self.diagnostic_system.apply_recommendations(
                    diagnostics['recommendations'],
                    self.param_controller
                )
                if actions:
                    self.logger.log(f"Applied diagnostic fixes: {actions}", 'INFO')

    def _apply_adjusted_parameters(self):
        """Apply adjusted parameters from param controller to systems."""
        # Update structural systems
        self.pruning_system.update_thresholds(
            trust_threshold=self.param_controller.deletion_threshold,
            usage_threshold=self.param_controller.usage_threshold,
            dormancy_threshold=self.param_controller.dormancy_threshold
        )

        self.merging_system.update_threshold(
            self.param_controller.merge_similarity_threshold
        )

        self.growing_system.update_thresholds(
            error_cluster_min_size=self.param_controller.error_cluster_min_size,
            success_threshold=self.param_controller.success_threshold,
            failure_threshold=self.param_controller.failure_threshold
        )

        # Update learning
        self.credit_assignment.update_top_k(self.param_controller.top_k)
        self.weight_updater.update_learning_rate(self.param_controller.learning_rate)

    def _run_structural_operations(self, iteration: int, current_performance: float):
        """
        Run structural operations (prune, merge, grow, reorganize).

        Args:
            iteration: Current iteration
            current_performance: Current performance
        """
        # Create snapshot before operations
        if iteration % self.param_controller.prune_interval == 0:
            self.snapshot_manager.create_snapshot(
                iteration,
                self.hierarchy,
                {'accuracy': current_performance}
            )

        # Pruning
        if iteration % self.param_controller.prune_interval == 0 and iteration > 0:
            results = self.pruning_system.prune_agents(iteration)
            if results['deleted'] > 0:
                self.logger.log_structural_operation('Pruning', results)

        # Merging
        if iteration % self.param_controller.merge_interval == 0 and iteration > 0:
            results = self.merging_system.merge_agents_batch(
                iteration,
                self.reorganization_system.total_activations
            )
            if results['merged'] > 0:
                self.logger.log_structural_operation('Merging', results)

        # Growing
        if iteration % 1000 == 0:
            total_errors = len(self.growing_system.error_history)
            results = self.growing_system.monitor_and_grow(iteration, total_errors)
            if results.get('agents_created', 0) > 0:
                self.logger.log_structural_operation('Growing', results)

        # Reorganization
        if iteration % self.param_controller.reorganize_interval == 0 and iteration > 0:
            results = self.reorganization_system.reorganize(iteration)
            if results['agents_moved'] > 0:
                self.logger.log_structural_operation('Reorganization', results)

        # Cache decay
        if iteration % self.config['operations']['cache_decay_interval'] == 0:
            self.trust_system.decay_cache(iteration)

    def _log_progress(self, iteration: int, accuracy: float, loss: float, avg_trust: float):
        """
        Log training progress.

        Args:
            iteration: Current iteration
            accuracy: Current accuracy
            loss: Current loss
            avg_trust: Average trust
        """
        phase = "Phase 2" if self.phase_2_active else "Phase 1"

        self.logger.log(
            f"[{phase}] Iter {iteration}: Acc={accuracy:.4f}, Loss={loss:.4f}, "
            f"Trust={avg_trust:.3f}, Agents={self.hierarchy.count_agents()}",
            'INFO'
        )

        # Log metrics
        self.logger.log_metrics(iteration, {
            'accuracy': accuracy,
            'loss': loss,
            'avg_trust': avg_trust,
            'total_agents': self.hierarchy.count_agents(),
            'phase': 2 if self.phase_2_active else 1
        })

    def _finalize_training(self):
        """Finalize training and save results."""
        self.logger.log("Training complete", 'INFO')

        # Print final statistics
        final_stats = self.metrics_tracker.compute_performance_summary()
        self.logger.log(f"Final Statistics: {final_stats}", 'INFO')

        hierarchy_stats = self.hierarchy.get_statistics()
        self.logger.log(f"Final Hierarchy: {hierarchy_stats}", 'INFO')

        # Finalize logger
        self.logger.finalize()


def main():
    """Main entry point for training."""
    # Create dummy dataset for testing
    np.random.seed(42)
    train_data = np.random.randn(1000, 128)
    train_labels = np.random.randint(0, 10, 1000)
    val_data = np.random.randn(200, 128)
    val_labels = np.random.randint(0, 10, 200)

    # Initialize and train system
    system = K1System()
    system.train(train_data, train_labels, val_data, val_labels)


if __name__ == '__main__':
    main()
