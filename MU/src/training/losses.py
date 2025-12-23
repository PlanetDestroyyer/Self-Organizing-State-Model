"""
Loss functions for MU Transformer training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LanguageModelingLoss(nn.Module):
    """
    Cross-entropy loss for language modeling

    Args:
        ignore_index: Index to ignore in loss computation (e.g., padding)
        label_smoothing: Label smoothing factor
    """

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, T, vocab_size]
            labels: [B, T]

        Returns:
            Scalar loss
        """
        # Reshape for cross entropy
        logits_flat = logits.view(-1, logits.size(-1))  # [B*T, vocab_size]
        labels_flat = labels.view(-1)  # [B*T]

        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )

        return loss


class InvarianceLoss(nn.Module):
    """
    Loss to encourage invariance in specific MU slots

    This encourages Identity and Invariant slots to remain stable.

    Args:
        invariant_positions: List of (row, col) tuples for invariant positions
        lambda_inv: Weight for invariance loss
    """

    def __init__(
        self,
        invariant_positions: list = [(0, 0), (0, 1), (0, 2), (3, 3)],
        lambda_inv: float = 1.0
    ):
        super().__init__()
        self.invariant_positions = invariant_positions
        self.lambda_inv = lambda_inv

    def forward(self, MU: torch.Tensor) -> torch.Tensor:
        """
        Compute variance of invariant slots across sequence

        Args:
            MU: [B, T, r, c] tensor of MU states

        Returns:
            Scalar loss (variance of invariant slots)
        """
        B, T, r, c = MU.shape

        total_variance = 0.0
        for row, col in self.invariant_positions:
            slot_values = MU[:, :, row, col]  # [B, T]
            # Compute variance across time dimension
            variance = slot_values.var(dim=1).mean()  # Mean over batch
            total_variance += variance

        # Normalize by number of positions
        loss = self.lambda_inv * total_variance / len(self.invariant_positions)

        return loss


class MUTransformerLoss(nn.Module):
    """
    Combined loss for MU Transformer

    Args:
        lambda_lm: Weight for language modeling loss
        lambda_inv: Weight for invariance loss
        ignore_index: Index to ignore in LM loss
        invariant_positions: Positions that should be invariant
    """

    def __init__(
        self,
        lambda_lm: float = 1.0,
        lambda_inv: float = 1.0,
        ignore_index: int = -100,
        invariant_positions: list = [(0, 0), (0, 1), (0, 2), (3, 3)]
    ):
        super().__init__()

        self.lambda_lm = lambda_lm
        self.lambda_inv = lambda_inv

        self.lm_loss = LanguageModelingLoss(ignore_index=ignore_index)
        self.inv_loss = InvarianceLoss(
            invariant_positions=invariant_positions,
            lambda_inv=lambda_inv
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        MU: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss

        Args:
            logits: [B, T, vocab_size]
            labels: [B, T]
            MU: Optional [B, T, r, c] MU states

        Returns:
            total_loss: Scalar total loss
            loss_dict: Dictionary of individual losses
        """
        # Language modeling loss
        lm_loss = self.lm_loss(logits, labels)

        # Invariance loss (if MU provided)
        if MU is not None and self.lambda_inv > 0:
            inv_loss = self.inv_loss(MU)
        else:
            inv_loss = torch.tensor(0.0, device=logits.device)

        # Total loss
        total_loss = self.lambda_lm * lm_loss + inv_loss

        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'lm': lm_loss.item(),
            'invariance': inv_loss.item() if isinstance(inv_loss, torch.Tensor) else 0.0
        }

        return total_loss, loss_dict


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning stable representations

    Used to encourage similar representations for augmented versions
    of the same input.

    Args:
        temperature: Temperature parameter for contrastive loss
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of embeddings

        Args:
            embeddings1: [B, D]
            embeddings2: [B, D]

        Returns:
            Scalar loss
        """
        B = embeddings1.size(0)

        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=-1)
        embeddings2 = F.normalize(embeddings2, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings1, embeddings2.T) / self.temperature  # [B, B]

        # Labels: diagonal elements are positives
        labels = torch.arange(B, device=embeddings1.device)

        # Contrastive loss (symmetric)
        loss1 = F.cross_entropy(similarity, labels)
        loss2 = F.cross_entropy(similarity.T, labels)

        loss = (loss1 + loss2) / 2

        return loss
