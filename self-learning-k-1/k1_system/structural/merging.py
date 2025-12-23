"""
Self-merging system for the Self-Learning K-1 System.

Combines redundant agents to reduce complexity while maintaining performance.
"""

import torch
from typing import List, Tuple
import numpy as np
from itertools import combinations
from ..core.agent import Agent
from ..core.hierarchy import Hierarchy


class MergingSystem:
    """
    Manages agent merging based on similarity and redundancy.
    """

    def __init__(self,
                 hierarchy: Hierarchy,
                 similarity_threshold: float = 0.85,
                 min_trust: float = 0.3,
                 validation_drop_threshold: float = 0.05):
        """
        Initialize merging system.

        Args:
            hierarchy: Hierarchy to operate on
            similarity_threshold: Minimum similarity to merge
            min_trust: Minimum trust for both agents
            validation_drop_threshold: Max performance drop allowed
        """
        self.hierarchy = hierarchy
        self.similarity_threshold = similarity_threshold
        self.min_trust = min_trust
        self.validation_drop_threshold = validation_drop_threshold

        self.merging_history = []
        self.coactivation_counts = {}  # Track how often agents activate together

    def record_coactivation(self, agent_ids: List[str]):
        """
        Record that a set of agents were activated together.

        Args:
            agent_ids: List of agent IDs that were activated
        """
        # Record all pairs
        for id1, id2 in combinations(sorted(agent_ids), 2):
            key = (id1, id2)
            self.coactivation_counts[key] = self.coactivation_counts.get(key, 0) + 1

    def compute_coactivation_similarity(self,
                                       agent_a: Agent,
                                       agent_b: Agent,
                                       total_activations: int) -> float:
        """
        Compute similarity based on coactivation frequency.

        Args:
            agent_a: First agent
            agent_b: Second agent
            total_activations: Total number of activations tracked

        Returns:
            Coactivation similarity (0.0 to 1.0)
        """
        if total_activations == 0:
            return 0.0

        key = tuple(sorted([agent_a.id, agent_b.id]))
        coactivations = self.coactivation_counts.get(key, 0)

        return coactivations / total_activations

    def compute_output_correlation(self,
                                   agent_a: Agent,
                                   agent_b: Agent,
                                   sample_inputs: np.ndarray) -> float:
        """
        Compute correlation between agent outputs.

        Args:
            agent_a: First agent
            agent_b: Second agent
            sample_inputs: Sample inputs to test (n_samples x input_dim)

        Returns:
            Output correlation (-1.0 to 1.0)
        """
        if sample_inputs is None or len(sample_inputs) == 0:
            return 0.0

        outputs_a = []
        outputs_b = []

        for x in sample_inputs:
            out_a = agent_a.forward(x)
            out_b = agent_b.forward(x)
            outputs_a.append(out_a)
            outputs_b.append(out_b)

        # Flatten outputs
        outputs_a = np.array(outputs_a).flatten()
        outputs_b = np.array(outputs_b).flatten()

        # Compute correlation
        if np.std(outputs_a) == 0 or np.std(outputs_b) == 0:
            return 0.0

        correlation = np.corrcoef(outputs_a, outputs_b)[0, 1]
        return correlation

    def compute_trust_similarity(self, agent_a: Agent, agent_b: Agent) -> float:
        """
        Compute similarity based on trust scores.

        Args:
            agent_a: First agent
            agent_b: Second agent

        Returns:
            Trust similarity (0.0 to 1.0)
        """
        return 1.0 - abs(agent_a.trust - agent_b.trust)

    def compute_similarity(self,
                          agent_a: Agent,
                          agent_b: Agent,
                          total_activations: int,
                          sample_inputs: np.ndarray = None) -> float:
        """
        Compute overall similarity between two agents.

        Args:
            agent_a: First agent
            agent_b: Second agent
            total_activations: Total activations for coactivation calculation
            sample_inputs: Optional sample inputs for output correlation

        Returns:
            Overall similarity score (0.0 to 1.0)
        """
        # Coactivation similarity
        coactivation_sim = self.compute_coactivation_similarity(
            agent_a, agent_b, total_activations
        )

        # Output correlation (if sample inputs provided)
        if sample_inputs is not None and len(sample_inputs) > 0:
            output_corr = self.compute_output_correlation(agent_a, agent_b, sample_inputs)
            output_sim = (output_corr + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
        else:
            output_sim = 0.0

        # Trust similarity
        trust_sim = self.compute_trust_similarity(agent_a, agent_b)

        # Weighted combination
        if sample_inputs is not None and len(sample_inputs) > 0:
            similarity = 0.4 * coactivation_sim + 0.4 * output_sim + 0.2 * trust_sim
        else:
            similarity = 0.5 * coactivation_sim + 0.5 * trust_sim

        return similarity

    def find_merge_candidates(self,
                             total_activations: int,
                             sample_inputs: np.ndarray = None) -> List[Tuple[Agent, Agent, float]]:
        """
        Find pairs of agents that are candidates for merging.

        Args:
            total_activations: Total activations for similarity calculation
            sample_inputs: Optional sample inputs

        Returns:
            List of (agent_a, agent_b, similarity) tuples
        """
        candidates = []
        all_agents = self.hierarchy.get_all_agents()

        # Only consider agents with same parent (siblings)
        for agent in all_agents:
            if agent.parent is None:
                continue

            siblings = self.hierarchy.get_siblings(agent)

            for sibling in siblings:
                # Skip if already processed this pair
                if (sibling, agent, 0) in [(c[1], c[0], 0) for c in candidates]:
                    continue

                # Check minimum trust
                if agent.trust < self.min_trust or sibling.trust < self.min_trust:
                    continue

                # Compute similarity
                similarity = self.compute_similarity(
                    agent, sibling, total_activations, sample_inputs
                )

                # Check threshold
                if similarity >= self.similarity_threshold:
                    candidates.append((agent, sibling, similarity))

        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)

        return candidates

    def merge_agents(self,
                    agent_a: Agent,
                    agent_b: Agent,
                    current_iteration: int) -> Agent:
        """
        Merge two PyTorch agents by averaging parameters.

        Args:
            agent_a: First agent
            agent_b: Second agent
            current_iteration: Current iteration

        Returns:
            Merged agent
        """
        # Create merged agent
        merged = Agent(
            agent_id=f"merged_{agent_a.id}_{agent_b.id}",
            agent_type=agent_a.agent_type,
            specialty=f"{agent_a.specialty} + {agent_b.specialty}",
            input_dim=agent_a.input_dim,
            hidden_dim=agent_a.hidden_dim,
            output_dim=agent_a.output_dim,
            initial_trust=max(agent_a.trust, agent_b.trust),
            creation_iteration=current_iteration
        ).to(agent_a.device)

        # Average PyTorch parameters
        with torch.no_grad():
            for (name_a, param_a), (name_b, param_b) in zip(
                agent_a.named_parameters(), agent_b.named_parameters()
            ):
                # Get corresponding parameter in merged agent
                merged_param = dict(merged.named_parameters())[name_a]
                # Average the two parameters
                merged_param.data = 0.5 * param_a.data + 0.5 * param_b.data

        # Combine performance metrics
        merged.success_count = agent_a.success_count + agent_b.success_count
        merged.failure_count = agent_a.failure_count + agent_b.failure_count
        merged.total_error_reduction = agent_a.total_error_reduction + agent_b.total_error_reduction
        merged.usage_count = agent_a.usage_count + agent_b.usage_count

        return merged

    def validate_merge(self,
                      agent_a: Agent,
                      agent_b: Agent,
                      merged: Agent,
                      validation_func,
                      baseline_perf: float) -> bool:
        """
        Validate that merging doesn't hurt performance.

        Args:
            agent_a: First agent to merge
            agent_b: Second agent to merge
            merged: Merged agent
            validation_func: Validation performance function
            baseline_perf: Baseline performance

        Returns:
            True if merge is safe
        """
        # Replace agents with merged agent
        parent = agent_a.parent

        # Transfer all children
        for child in list(agent_a.child_agents):
            agent_a.remove_child(child)
            merged.add_child(child)

        for child in list(agent_b.child_agents):
            agent_b.remove_child(child)
            merged.add_child(child)

        # Remove old agents and add merged
        if parent:
            parent.remove_child(agent_a)
            parent.remove_child(agent_b)
            parent.add_child(merged)

        # Validate
        new_perf = validation_func()

        # Compute performance drop
        perf_drop = (baseline_perf - new_perf) / baseline_perf if baseline_perf > 0 else 0

        # Restore old configuration
        if parent:
            parent.remove_child(merged)
            parent.add_child(agent_a)
            parent.add_child(agent_b)

        for child in list(merged.child_agents):
            merged.remove_child(child)

        # Restore original children
        for child in agent_a.child_agents:
            agent_a.add_child(child)
        for child in agent_b.child_agents:
            agent_b.add_child(child)

        return perf_drop <= self.validation_drop_threshold

    def execute_merge(self,
                     agent_a: Agent,
                     agent_b: Agent,
                     merged: Agent,
                     current_iteration: int):
        """
        Permanently execute a merge.

        Args:
            agent_a: First agent
            agent_b: Second agent
            merged: Merged agent
            current_iteration: Current iteration
        """
        parent = agent_a.parent

        # Transfer children
        for child in list(agent_a.child_agents):
            agent_a.remove_child(child)
            merged.add_child(child)

        for child in list(agent_b.child_agents):
            agent_b.remove_child(child)
            merged.add_child(child)

        # Remove old agents
        self.hierarchy.remove_agent(agent_a)
        self.hierarchy.remove_agent(agent_b)

        # Clean up optimizer state for removed agents (PyTorch)
        if hasattr(self, 'weight_updater') and hasattr(self.weight_updater, 'remove_agent_optimizer'):
            self.weight_updater.remove_agent_optimizer(agent_a.id)
            self.weight_updater.remove_agent_optimizer(agent_b.id)

        # Add merged agent
        if parent:
            self.hierarchy.add_agent(merged, parent)

        # Log merge
        self.merging_history.append({
            'iteration': current_iteration,
            'agent_a_id': agent_a.id,
            'agent_b_id': agent_b.id,
            'merged_id': merged.id,
            'agent_a_trust': agent_a.trust,
            'agent_b_trust': agent_b.trust,
            'merged_trust': merged.trust
        })

    def merge_agents_batch(self,
                          current_iteration: int,
                          total_activations: int,
                          sample_inputs: np.ndarray = None,
                          validation_func=None,
                          baseline_perf: float = None) -> dict:
        """
        Execute merging cycle.

        Args:
            current_iteration: Current iteration
            total_activations: Total activations
            sample_inputs: Sample inputs for similarity
            validation_func: Optional validation function
            baseline_perf: Optional baseline performance

        Returns:
            Dictionary with merging statistics
        """
        # Find candidates
        candidates = self.find_merge_candidates(total_activations, sample_inputs)

        merged_count = 0
        failed_validation = 0
        merged_pairs = []

        for agent_a, agent_b, similarity in candidates:
            # Skip if either agent was already merged
            if agent_a.id in [p[0] for p in merged_pairs]:
                continue
            if agent_b.id in [p[1] for p in merged_pairs]:
                continue

            # Create merged agent
            merged = self.merge_agents(agent_a, agent_b, current_iteration)

            # Validate if function provided
            if validation_func and baseline_perf is not None:
                if not self.validate_merge(agent_a, agent_b, merged, validation_func, baseline_perf):
                    failed_validation += 1
                    continue

            # Execute merge
            self.execute_merge(agent_a, agent_b, merged, current_iteration)
            merged_count += 1
            merged_pairs.append((agent_a.id, agent_b.id))

        return {
            'candidates': len(candidates),
            'merged': merged_count,
            'failed_validation': failed_validation,
            'merged_pairs': merged_pairs
        }

    def update_threshold(self, new_threshold: float):
        """
        Update similarity threshold.

        Args:
            new_threshold: New threshold value
        """
        self.similarity_threshold = max(0.0, min(1.0, new_threshold))

    def get_merging_statistics(self) -> dict:
        """
        Get statistics about merging history.

        Returns:
            Dictionary of statistics
        """
        if not self.merging_history:
            return {
                'total_merges': 0
            }

        return {
            'total_merges': len(self.merging_history),
            'recent_merges': len([e for e in self.merging_history[-100:]])
        }
