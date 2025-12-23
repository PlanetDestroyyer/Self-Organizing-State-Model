"""
Evaluation metrics for MU Transformer
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@torch.no_grad()
def compute_perplexity(
    model,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> float:
    """
    Compute perplexity on a dataset

    Args:
        model: Language model
        dataloader: Dataloader
        device: Device to use

    Returns:
        Perplexity value
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits, _ = model(input_ids)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='sum'
        )

        # Count non-padding tokens
        mask = labels != -100
        num_tokens = mask.sum().item()

        total_loss += loss.item()
        total_tokens += num_tokens

    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))  # Cap for numerical stability

    return perplexity


@torch.no_grad()
def compute_wic_accuracy(
    model,
    dataloader: DataLoader,
    device: str = 'cuda',
    pooling: str = 'mean'
) -> Dict:
    """
    Evaluate on Word-in-Context (WiC) task

    This requires training a simple probe on top of MU representations.

    Args:
        model: Model to evaluate
        dataloader: WiC dataloader
        device: Device to use
        pooling: Pooling strategy ('mean', 'max', 'cls')

    Returns:
        Dictionary with accuracy and other metrics
    """
    model.eval()

    # Collect representations and labels
    all_embeddings1 = []
    all_embeddings2 = []
    all_labels = []

    for batch in dataloader:
        sent1_ids = batch['sent1_input_ids'].to(device)
        sent2_ids = batch['sent2_input_ids'].to(device)
        labels = batch['label']

        # Get representations
        _, mu1 = model(sent1_ids)
        _, mu2 = model(sent2_ids)

        # Pool representations
        if pooling == 'mean':
            emb1 = mu1.mean(dim=1).flatten(start_dim=1)  # [B, r*c]
            emb2 = mu2.mean(dim=1).flatten(start_dim=1)
        elif pooling == 'max':
            emb1 = mu1.max(dim=1)[0].flatten(start_dim=1)
            emb2 = mu2.max(dim=1)[0].flatten(start_dim=1)
        else:  # 'cls' - use first token
            emb1 = mu1[:, 0].flatten(start_dim=1)
            emb2 = mu2[:, 0].flatten(start_dim=1)

        all_embeddings1.append(emb1.cpu())
        all_embeddings2.append(emb2.cpu())
        all_labels.append(labels)

    # Concatenate
    all_embeddings1 = torch.cat(all_embeddings1, dim=0)
    all_embeddings2 = torch.cat(all_embeddings2, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Train simple probe (logistic regression)
    # Use cosine similarity as feature
    similarities = F.cosine_similarity(all_embeddings1, all_embeddings2, dim=-1)

    # Threshold at 0.5 (median)
    threshold = similarities.median().item()
    predictions = (similarities > threshold).long()

    # Compute metrics
    accuracy = accuracy_score(all_labels.numpy(), predictions.numpy())

    return {
        'accuracy': accuracy,
        'threshold': threshold
    }


@torch.no_grad()
def compute_embedding_stability(
    model,
    original_inputs: torch.Tensor,
    augmented_inputs: torch.Tensor,
    device: str = 'cuda'
) -> Dict:
    """
    Compute embedding stability under augmentation

    Args:
        model: Model to evaluate
        original_inputs: Original input IDs [B, T]
        augmented_inputs: Augmented input IDs [B, T]
        device: Device to use

    Returns:
        Dictionary with stability metrics
    """
    model.eval()

    original_inputs = original_inputs.to(device)
    augmented_inputs = augmented_inputs.to(device)

    # Get MU representations
    _, mu_orig = model(original_inputs)
    _, mu_aug = model(augmented_inputs)

    # Pool to get sentence representations
    emb_orig = mu_orig.mean(dim=1).flatten(start_dim=1)  # [B, r*c]
    emb_aug = mu_aug.mean(dim=1).flatten(start_dim=1)

    # Compute cosine similarities
    similarities = F.cosine_similarity(emb_orig, emb_aug, dim=-1)

    # Compute statistics
    mean_similarity = similarities.mean().item()
    std_similarity = similarities.std().item()
    min_similarity = similarities.min().item()
    max_similarity = similarities.max().item()

    return {
        'mean': mean_similarity,
        'std': std_similarity,
        'min': min_similarity,
        'max': max_similarity
    }


@torch.no_grad()
def analyze_slot_specialization(
    model,
    dataloader: DataLoader,
    device: str = 'cuda',
    max_batches: Optional[int] = None
) -> Dict:
    """
    Analyze how different slots specialize

    Compute variance of each slot position across contexts.
    High variance = context-sensitive
    Low variance = stable/invariant

    Args:
        model: Model to evaluate
        dataloader: Dataloader
        device: Device to use
        max_batches: Maximum number of batches to process

    Returns:
        Dictionary with slot variance analysis
    """
    model.eval()

    all_mus = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)

        # Get MU representations
        _, MU = model(input_ids)

        all_mus.append(MU.cpu())

    # Concatenate
    all_mus = torch.cat(all_mus, dim=0)  # [N, T, r, c]

    # Compute variance per slot position across all samples and time steps
    variances = all_mus.var(dim=[0, 1])  # [r, c]

    # Compute mean values per slot
    means = all_mus.mean(dim=[0, 1])  # [r, c]

    # Identify which slots are most/least variable
    variance_flat = variances.flatten()
    most_variable_idx = variance_flat.argmax().item()
    least_variable_idx = variance_flat.argmin().item()

    r, c = variances.shape
    most_variable_pos = (most_variable_idx // c, most_variable_idx % c)
    least_variable_pos = (least_variable_idx // c, least_variable_idx % c)

    return {
        'variances': variances.numpy(),
        'means': means.numpy(),
        'most_variable': {
            'position': most_variable_pos,
            'variance': variance_flat[most_variable_idx].item()
        },
        'least_variable': {
            'position': least_variable_pos,
            'variance': variance_flat[least_variable_idx].item()
        }
    }


def compute_retrieval_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1
) -> Dict:
    """
    Compute retrieval metrics (Recall@K)

    Args:
        embeddings: Embeddings [N, D]
        labels: Labels [N]
        k: K for Recall@K

    Returns:
        Dictionary with retrieval metrics
    """
    N = embeddings.size(0)

    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1)

    # Compute similarity matrix
    similarities = torch.matmul(embeddings, embeddings.T)  # [N, N]

    # For each query, find top-k most similar (excluding itself)
    similarities.fill_diagonal_(-float('inf'))  # Exclude self
    topk_indices = similarities.topk(k, dim=-1).indices  # [N, k]

    # Check if any of top-k have same label
    topk_labels = labels[topk_indices]  # [N, k]
    query_labels = labels.unsqueeze(1).expand(-1, k)  # [N, k]

    matches = (topk_labels == query_labels).any(dim=-1)  # [N]
    recall_at_k = matches.float().mean().item()

    return {
        f'recall@{k}': recall_at_k
    }
