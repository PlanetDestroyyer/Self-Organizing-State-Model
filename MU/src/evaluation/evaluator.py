"""
Comprehensive evaluator for MU Transformer
"""
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional
import numpy as np
from pathlib import Path

from .metrics import (
    compute_perplexity,
    compute_wic_accuracy,
    compute_embedding_stability,
    analyze_slot_specialization,
    compute_retrieval_metrics
)
from ..data.datasets import get_dataloaders, WikiTextDataset, WiCDataset
from ..data.augmentation import TextAugmenter, create_augmented_pairs
from ..utils.logging_utils import setup_logger


class Evaluator:
    """
    Comprehensive evaluator for MU Transformer

    Args:
        model: Model to evaluate
        device: Device to use for evaluation
        data_config: Data configuration dictionary
    """

    def __init__(
        self,
        model,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        data_config: Optional[Dict] = None
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.data_config = data_config or {
            'dataset': 'wikitext-2',
            'sequence_length': 256,
            'batch_size': 32,
            'num_workers': 4,
            'vocab_size': 30000
        }

        self.logger = setup_logger(name='evaluator')

    def evaluate_language_modeling(
        self,
        test_loader: Optional[DataLoader] = None
    ) -> Dict:
        """
        Evaluate language modeling performance

        Args:
            test_loader: Optional test dataloader

        Returns:
            Dictionary with LM metrics
        """
        self.logger.info("Evaluating language modeling...")

        if test_loader is None:
            # Load test data
            config = {**self.data_config, 'dataset': 'wikitext-2'}
            _, _, test_loader = get_dataloaders(config, test_mode=False)

        perplexity = compute_perplexity(self.model, test_loader, self.device)

        self.logger.info(f"Language Modeling Perplexity: {perplexity:.2f}")

        return {
            'perplexity': perplexity
        }

    def evaluate_wic(self) -> Dict:
        """
        Evaluate on Word-in-Context task

        Returns:
            Dictionary with WiC metrics
        """
        self.logger.info("Evaluating WiC (Word Sense Disambiguation)...")

        # Load WiC dataset
        config = {**self.data_config, 'dataset': 'wic'}
        try:
            _, val_loader, _ = get_dataloaders(config, test_mode=False)

            metrics = compute_wic_accuracy(
                self.model,
                val_loader,
                self.device,
                pooling='mean'
            )

            self.logger.info(f"WiC Accuracy: {metrics['accuracy']:.4f}")

            return metrics

        except Exception as e:
            self.logger.warning(f"WiC evaluation failed: {e}")
            return {'accuracy': 0.0}

    def evaluate_stability(
        self,
        num_samples: int = 100
    ) -> Dict:
        """
        Evaluate embedding stability under augmentation

        Args:
            num_samples: Number of samples to test

        Returns:
            Dictionary with stability metrics
        """
        self.logger.info("Evaluating embedding stability...")

        # Load test data
        config = {**self.data_config, 'dataset': 'wikitext-2'}
        _, _, test_loader = get_dataloaders(config, test_mode=True)

        # Get samples
        samples = []
        for batch in test_loader:
            samples.append(batch['input_ids'])
            if len(samples) * batch['input_ids'].size(0) >= num_samples:
                break

        original_inputs = torch.cat(samples, dim=0)[:num_samples]

        # Create augmented versions
        augmenter = TextAugmenter(dropout_prob=0.1, swap_prob=0.1)
        augmented_inputs = augmenter.augment(original_inputs, methods=['dropout', 'swap'])

        # Compute stability
        stability_metrics = compute_embedding_stability(
            self.model,
            original_inputs,
            augmented_inputs,
            self.device
        )

        self.logger.info(
            f"Embedding Stability: {stability_metrics['mean']:.4f} ± {stability_metrics['std']:.4f}"
        )

        return stability_metrics

    def analyze_slot_usage(
        self,
        max_batches: int = 50
    ) -> Dict:
        """
        Analyze how different MU slots are used

        Args:
            max_batches: Maximum batches to analyze

        Returns:
            Dictionary with slot analysis
        """
        self.logger.info("Analyzing slot usage...")

        # Check if model has MU structure
        if not hasattr(self.model, 'model'):
            self.logger.warning("Model doesn't appear to be MU Transformer, skipping slot analysis")
            return {}

        # Load test data
        config = {**self.data_config, 'dataset': 'wikitext-2'}
        _, _, test_loader = get_dataloaders(config, test_mode=True)

        # Analyze slots
        slot_metrics = analyze_slot_specialization(
            self.model,
            test_loader,
            self.device,
            max_batches=max_batches
        )

        self.logger.info(f"Most variable slot: {slot_metrics['most_variable']['position']}")
        self.logger.info(f"Least variable slot: {slot_metrics['least_variable']['position']}")

        return slot_metrics

    def evaluate_all(
        self,
        tasks: Optional[list] = None
    ) -> Dict:
        """
        Run all evaluations

        Args:
            tasks: List of tasks to evaluate (if None, run all)

        Returns:
            Dictionary with all metrics
        """
        if tasks is None:
            tasks = ['lm', 'wic', 'stability', 'slots']

        results = {}

        if 'lm' in tasks:
            results['language_modeling'] = self.evaluate_language_modeling()

        if 'wic' in tasks:
            results['wic'] = self.evaluate_wic()

        if 'stability' in tasks:
            results['stability'] = self.evaluate_stability()

        if 'slots' in tasks:
            results['slot_analysis'] = self.analyze_slot_usage()

        return results

    def save_results(
        self,
        results: Dict,
        output_file: str = 'results/summary.txt'
    ):
        """
        Save evaluation results to file

        Args:
            results: Results dictionary
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MU TRANSFORMER EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")

            # Language modeling
            if 'language_modeling' in results:
                lm = results['language_modeling']
                f.write("Language Modeling:\n")
                f.write(f"  Perplexity: {lm['perplexity']:.2f}\n\n")

            # WiC
            if 'wic' in results:
                wic = results['wic']
                f.write("Word-in-Context (WSD):\n")
                f.write(f"  Accuracy: {wic['accuracy']:.4f}\n\n")

            # Stability
            if 'stability' in results:
                stab = results['stability']
                f.write("Embedding Stability:\n")
                f.write(f"  Mean similarity: {stab['mean']:.4f}\n")
                f.write(f"  Std similarity: {stab['std']:.4f}\n\n")

            # Slot analysis
            if 'slot_analysis' in results:
                slots = results['slot_analysis']
                if slots:
                    f.write("Slot Specialization:\n")
                    f.write(f"  Most variable: {slots['most_variable']['position']} "
                           f"(var={slots['most_variable']['variance']:.4f})\n")
                    f.write(f"  Least variable: {slots['least_variable']['position']} "
                           f"(var={slots['least_variable']['variance']:.4f})\n")

        self.logger.info(f"Results saved to {output_path}")
