"""
Self-growing system for the Self-Learning K-1 System.

Creates new specialist agents to fill knowledge gaps.
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from ..core.agent import Agent
from ..core.hierarchy import Hierarchy


class ErrorCluster:
    """
    Represents a cluster of similar errors.
    """

    def __init__(self, cluster_id: int):
        self.cluster_id = cluster_id
        self.errors = []
        self.error_vectors = []
        self.frequency = 0
        self.avg_magnitude = 0.0
        self.persistence_count = 0
        self.first_seen = None
        self.last_seen = None

    def add_error(self, error_vector: np.ndarray, magnitude: float, iteration: int):
        """Add an error to this cluster."""
        self.errors.append({
            'vector': error_vector,
            'magnitude': magnitude,
            'iteration': iteration
        })
        self.error_vectors.append(error_vector)
        self.last_seen = iteration

        if self.first_seen is None:
            self.first_seen = iteration

        self.frequency += 1
        self.avg_magnitude = np.mean([e['magnitude'] for e in self.errors])

    def get_centroid(self) -> np.ndarray:
        """Get centroid of error vectors."""
        if not self.error_vectors:
            return np.zeros(1)
        return np.mean(self.error_vectors, axis=0)

    def persists_for(self, current_iteration: int) -> int:
        """Get how long this cluster has persisted."""
        if self.first_seen is None:
            return 0
        return current_iteration - self.first_seen


class GrowingSystem:
    """
    Manages creation of new specialist agents for knowledge gaps.
    """

    def __init__(self,
                 hierarchy: Hierarchy,
                 error_cluster_min_size: int = 100,
                 error_frequency_threshold: float = 0.01,
                 new_agent_initial_trust: float = 0.3,
                 validation_period: int = 1000,
                 success_threshold: float = 0.2,
                 failure_threshold: float = 0.05,
                 max_agents: int = 1000):
        """
        Initialize growing system.

        Args:
            hierarchy: Hierarchy to grow
            error_cluster_min_size: Minimum errors in cluster
            error_frequency_threshold: Minimum error frequency
            new_agent_initial_trust: Initial trust for new agents
            validation_period: Iterations to validate new agent
            success_threshold: Error reduction needed for success
            failure_threshold: Max error reduction for failure
            max_agents: Maximum total agents allowed
        """
        self.hierarchy = hierarchy
        self.error_cluster_min_size = error_cluster_min_size
        self.error_frequency_threshold = error_frequency_threshold
        self.new_agent_initial_trust = new_agent_initial_trust
        self.validation_period = validation_period
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.max_agents = max_agents

        self.error_history = []
        self.error_clusters: Dict[int, ErrorCluster] = {}
        self.agent_creation_history = []
        self.agents_under_validation = []  # (agent, start_iteration, cluster_id)

    def record_error(self,
                    error_vector: np.ndarray,
                    magnitude: float,
                    activated_agents: List[Agent],
                    iteration: int):
        """
        Record an error for gap detection.

        Args:
            error_vector: Error vector
            magnitude: Error magnitude
            activated_agents: Agents that were activated
            iteration: Current iteration
        """
        self.error_history.append({
            'vector': error_vector,
            'magnitude': magnitude,
            'agents': [a.id for a in activated_agents],
            'best_agent_trust': max([a.trust for a in activated_agents]) if activated_agents else 0.0,
            'iteration': iteration
        })

    def cluster_errors(self, n_clusters: int = 10) -> Dict[int, ErrorCluster]:
        """
        Cluster recent errors to find patterns.

        Args:
            n_clusters: Number of clusters

        Returns:
            Dictionary of error clusters
        """
        if len(self.error_history) < self.error_cluster_min_size:
            return {}

        # Get recent errors
        recent_errors = self.error_history[-10000:]  # Last 10k errors

        # Extract error vectors
        error_vectors = np.array([e['vector'] for e in recent_errors])

        # Cluster using K-means
        kmeans = KMeans(n_clusters=min(n_clusters, len(recent_errors)), random_state=42)
        labels = kmeans.fit_predict(error_vectors)

        # Create error clusters
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = ErrorCluster(label)

            error = recent_errors[idx]
            clusters[label].add_error(
                error['vector'],
                error['magnitude'],
                error['iteration']
            )

        self.error_clusters = clusters
        return clusters

    def is_knowledge_gap(self, cluster: ErrorCluster, total_errors: int, current_iteration: int) -> bool:
        """
        Determine if an error cluster represents a knowledge gap.

        Args:
            cluster: Error cluster to check
            total_errors: Total number of errors
            current_iteration: Current iteration

        Returns:
            True if this is a knowledge gap
        """
        # Check all criteria
        criteria = []

        # Significant cluster size
        if cluster.frequency >= self.error_cluster_min_size:
            criteria.append(True)
        else:
            criteria.append(False)

        # High frequency relative to total errors
        if total_errors > 0 and (cluster.frequency / total_errors) >= self.error_frequency_threshold:
            criteria.append(True)
        else:
            criteria.append(False)

        # No agent handles it well (low best agent trust in errors)
        avg_best_trust = np.mean([e.get('best_agent_trust', 0) for e in cluster.errors])
        if avg_best_trust < 0.5:
            criteria.append(True)
        else:
            criteria.append(False)

        # Persistent over time
        if cluster.persists_for(current_iteration) > 5000:
            criteria.append(True)
        else:
            criteria.append(False)

        # All criteria must be met
        return all(criteria)

    def characterize_gap(self, cluster: ErrorCluster) -> str:
        """
        Characterize the knowledge gap to create specialty.

        Args:
            cluster: Error cluster

        Returns:
            Specialty string for new agent
        """
        # Simple characterization based on error patterns
        centroid = cluster.get_centroid()

        # For now, use cluster ID and characteristics
        specialty = f"ErrorPattern_{cluster.cluster_id}_Magnitude_{cluster.avg_magnitude:.2f}"

        return specialty

    def create_agent_for_gap(self,
                            cluster: ErrorCluster,
                            current_iteration: int) -> Agent:
        """
        Create a new specialist agent for a knowledge gap.

        Args:
            cluster: Error cluster representing the gap
            current_iteration: Current iteration

        Returns:
            New specialist agent
        """
        # Characterize the gap
        specialty = self.characterize_gap(cluster)

        # Find best parent in hierarchy
        best_parent = self.hierarchy.find_closest_manager(specialty)

        if best_parent is None:
            best_parent = self.hierarchy.root

        # Create new agent
        # Use same dimensions as parent
        new_agent = Agent(
            agent_id=f"specialist_{current_iteration}_{cluster.cluster_id}",
            agent_type='agent' if best_parent.agent_type == 'manager' else 'sub_agent',
            specialty=specialty,
            input_dim=best_parent.input_dim if hasattr(best_parent, 'input_dim') else 128,
            hidden_dim=best_parent.hidden_dim if hasattr(best_parent, 'hidden_dim') else 64,
            output_dim=best_parent.output_dim if hasattr(best_parent, 'output_dim') else 128,
            initial_trust=self.new_agent_initial_trust,
            creation_iteration=current_iteration
        )

        # Add to hierarchy
        self.hierarchy.add_agent(new_agent, best_parent)

        # Add to validation list
        self.agents_under_validation.append({
            'agent': new_agent,
            'start_iteration': current_iteration,
            'cluster_id': cluster.cluster_id,
            'initial_error_rate': cluster.avg_magnitude
        })

        # Log creation
        self.agent_creation_history.append({
            'iteration': current_iteration,
            'agent_id': new_agent.id,
            'specialty': specialty,
            'cluster_id': cluster.cluster_id,
            'cluster_size': cluster.frequency,
            'parent_id': best_parent.id
        })

        return new_agent

    def validate_new_agents(self, current_iteration: int) -> dict:
        """
        Validate agents under validation period.

        Args:
            current_iteration: Current iteration

        Returns:
            Dictionary with validation results
        """
        to_remove = []
        results = {
            'validated': [],
            'removed': [],
            'still_validating': []
        }

        for entry in self.agents_under_validation:
            agent = entry['agent']
            start_iteration = entry['start_iteration']
            cluster_id = entry['cluster_id']
            initial_error = entry['initial_error_rate']

            # Check if validation period complete
            if current_iteration - start_iteration < self.validation_period:
                results['still_validating'].append(agent.id)
                continue

            # Measure error reduction for this cluster
            # Get recent errors from this cluster
            cluster = self.error_clusters.get(cluster_id)

            if cluster:
                current_error = cluster.avg_magnitude
                error_reduction = (initial_error - current_error) / initial_error if initial_error > 0 else 0
            else:
                # Cluster disappeared - could be good or bad
                error_reduction = 0.5  # Assume moderate success

            # Decision
            if error_reduction > self.success_threshold:
                # Success! Keep agent and boost trust
                agent.trust = 0.5
                results['validated'].append(agent.id)
                to_remove.append(entry)

            elif error_reduction < self.failure_threshold:
                # Failed experiment, delete
                self.hierarchy.remove_agent(agent)
                results['removed'].append(agent.id)
                to_remove.append(entry)

            else:
                # Marginal, keep but continue monitoring
                results['validated'].append(agent.id)
                to_remove.append(entry)

        # Remove from validation list
        for entry in to_remove:
            self.agents_under_validation.remove(entry)

        return results

    def monitor_and_grow(self, current_iteration: int, total_errors: int) -> dict:
        """
        Monitor for knowledge gaps and create agents as needed.

        Args:
            current_iteration: Current iteration
            total_errors: Total number of errors

        Returns:
            Dictionary with growth statistics
        """
        results = {
            'gaps_identified': 0,
            'agents_created': 0,
            'agents_validated': 0,
            'agents_removed': 0
        }

        # Check if we can create more agents
        if self.hierarchy.count_agents() >= self.max_agents:
            results['at_capacity'] = True
            return results

        # Cluster errors
        if len(self.error_history) >= self.error_cluster_min_size:
            clusters = self.cluster_errors()

            # Identify knowledge gaps
            for cluster_id, cluster in clusters.items():
                if self.is_knowledge_gap(cluster, total_errors, current_iteration):
                    results['gaps_identified'] += 1

                    # Create new agent for this gap
                    new_agent = self.create_agent_for_gap(cluster, current_iteration)
                    results['agents_created'] += 1

        # Validate existing agents under validation
        validation_results = self.validate_new_agents(current_iteration)
        results['agents_validated'] = len(validation_results['validated'])
        results['agents_removed'] = len(validation_results['removed'])

        return results

    def update_thresholds(self,
                         error_cluster_min_size: int = None,
                         error_frequency_threshold: float = None,
                         success_threshold: float = None,
                         failure_threshold: float = None):
        """
        Update growth thresholds.

        Args:
            error_cluster_min_size: New minimum cluster size
            error_frequency_threshold: New frequency threshold
            success_threshold: New success threshold
            failure_threshold: New failure threshold
        """
        if error_cluster_min_size is not None:
            self.error_cluster_min_size = error_cluster_min_size
        if error_frequency_threshold is not None:
            self.error_frequency_threshold = error_frequency_threshold
        if success_threshold is not None:
            self.success_threshold = success_threshold
        if failure_threshold is not None:
            self.failure_threshold = failure_threshold

    def get_growth_statistics(self) -> dict:
        """
        Get statistics about agent creation.

        Returns:
            Dictionary of statistics
        """
        return {
            'total_created': len(self.agent_creation_history),
            'under_validation': len(self.agents_under_validation),
            'recent_creations': len([e for e in self.agent_creation_history[-100:]]),
            'error_history_size': len(self.error_history),
            'active_clusters': len(self.error_clusters)
        }
