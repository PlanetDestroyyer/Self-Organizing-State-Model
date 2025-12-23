"""Training and evaluation functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import math
from typing import Dict

from ..config import MUSOTAConfig

config = MUSOTAConfig()


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                scheduler, scaler: GradScaler, device: str, epoch: int, total_epochs: int) -> Dict:
    """Train for one epoch with mixed precision and gradient accumulation"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]")

    # Determine device type for autocast
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # Gradient accumulation setup
    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Mixed precision forward pass
        with autocast(device_type, enabled=config.use_mixed_precision):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward with gradient scaling
        scaler.scale(loss).backward()

        # Only update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # Metrics (multiply back by accumulation_steps for true loss)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_tokens += labels.numel()
            total_loss += loss.item() * accumulation_steps

        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}', 'acc': f'{correct/labels.numel():.4f}'})

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': total_correct / total_tokens
    }


def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> Dict:
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    # Determine device type for autocast
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with autocast(device_type, enabled=config.use_mixed_precision):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )

            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_tokens += labels.numel()
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 100))  # Cap to prevent overflow

    return {'loss': avg_loss, 'accuracy': accuracy, 'perplexity': perplexity}
