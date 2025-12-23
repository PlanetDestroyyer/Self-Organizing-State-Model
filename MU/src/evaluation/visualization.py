"""
Visualization utilities for MU Transformer analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List


def plot_training_curves(
    metrics: Dict,
    output_file: str = 'results/plots/training_curves.png'
):
    """
    Plot training curves

    Args:
        metrics: Dictionary with 'step', 'train_loss', 'val_loss', etc.
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    if 'step' in metrics and 'train_loss' in metrics:
        axes[0].plot(metrics['step'], metrics['train_loss'], label='Train Loss', alpha=0.7)
        if 'val_loss' in metrics:
            axes[0].plot(metrics['step'], metrics['val_loss'], label='Val Loss', alpha=0.7)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Perplexity curves
    if 'train_perplexity' in metrics:
        axes[1].plot(metrics['step'], metrics['train_perplexity'], label='Train PPL', alpha=0.7)
        if 'val_perplexity' in metrics:
            axes[1].plot(metrics['step'], metrics['val_perplexity'], label='Val PPL', alpha=0.7)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Perplexity')
        axes[1].set_title('Perplexity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to {output_path}")


def plot_slot_heatmap(
    variances: np.ndarray,
    output_file: str = 'results/plots/slot_heatmap.png',
    slot_labels: Optional[List[List[str]]] = None
):
    """
    Plot heatmap of slot variances

    Args:
        variances: [r, c] array of variances
        output_file: Output file path
        slot_labels: Optional labels for each slot
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if slot_labels is None:
        slot_labels = [
            ["I", "S1", "S2", "R1a"],
            ["R1b", "R2a", "R2b", "C1"],
            ["C2", "C3", "C4", "T1"],
            ["T2", "K1", "K2", "G1"]
        ]

    # Flatten labels for annotation
    annotations = np.array(slot_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        variances,
        annot=annotations,
        fmt='',
        cmap='YlOrRd',
        cbar_kws={'label': 'Variance'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('MU Slot Variance (Context Sensitivity)', fontsize=14)
    plt.xlabel('Column', fontsize=12)
    plt.ylabel('Row', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Slot heatmap saved to {output_path}")


def plot_stability_comparison(
    mu_stability: Dict,
    baseline_stability: Optional[Dict] = None,
    output_file: str = 'results/plots/stability_comparison.png'
):
    """
    Plot stability comparison between MU and baseline

    Args:
        mu_stability: MU model stability metrics
        baseline_stability: Baseline model stability metrics
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['MU Transformer']
    means = [mu_stability['mean']]
    stds = [mu_stability['std']]

    if baseline_stability is not None:
        models.append('Baseline')
        means.append(baseline_stability['mean'])
        stds.append(baseline_stability['std'])

    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e'][:len(models)])

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Cosine Similarity (Original vs Augmented)', fontsize=12)
    ax.set_title('Embedding Stability Under Augmentation', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Stability comparison saved to {output_path}")


def plot_results_summary(
    mu_results: Dict,
    baseline_results: Optional[Dict] = None,
    output_file: str = 'results/plots/results_summary.png'
):
    """
    Plot comprehensive results summary

    Args:
        mu_results: MU model results
        baseline_results: Baseline model results
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Perplexity
    if 'language_modeling' in mu_results:
        models = ['MU']
        ppls = [mu_results['language_modeling']['perplexity']]

        if baseline_results and 'language_modeling' in baseline_results:
            models.append('Baseline')
            ppls.append(baseline_results['language_modeling']['perplexity'])

        axes[0].bar(models, ppls, alpha=0.7, color=['#1f77b4', '#ff7f0e'][:len(models)])
        axes[0].set_ylabel('Perplexity')
        axes[0].set_title('Language Modeling')
        axes[0].grid(True, alpha=0.3, axis='y')

    # WiC Accuracy
    if 'wic' in mu_results:
        models = ['MU']
        accs = [mu_results['wic']['accuracy']]

        if baseline_results and 'wic' in baseline_results:
            models.append('Baseline')
            accs.append(baseline_results['wic']['accuracy'])

        axes[1].bar(models, accs, alpha=0.7, color=['#1f77b4', '#ff7f0e'][:len(models)])
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('WiC (Word Sense)')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3, axis='y')

    # Stability
    if 'stability' in mu_results:
        models = ['MU']
        stabs = [mu_results['stability']['mean']]

        if baseline_results and 'stability' in baseline_results:
            models.append('Baseline')
            stabs.append(baseline_results['stability']['mean'])

        axes[2].bar(models, stabs, alpha=0.7, color=['#1f77b4', '#ff7f0e'][:len(models)])
        axes[2].set_ylabel('Cosine Similarity')
        axes[2].set_title('Embedding Stability')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Results summary saved to {output_path}")
