"""
Self-Learning Time Embeddings for TEMPORAL Architecture
Time embeddings learn what "experience" means through gradients, not hardcoded rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Temporal_TimeEmbeddings(nn.Module):
    """
    Self-learning time embeddings that discover temporal patterns through gradients.

    Key Design:
    - Time embeddings are LEARNED, not hardcoded
    - Gradients flow through time just like content embeddings
    - Model discovers what "experience" means by minimizing prediction loss
    - Optional inference-time updates based on learned patterns
    """

    def __init__(self, vocab_size, time_dim, learning_mode='gradient'):
        super().__init__()
        self.vocab_size = vocab_size
        self.time_dim = time_dim
        self.learning_mode = learning_mode  # 'gradient' or 'hybrid'

        # Time embeddings - initialized to zeros, learns from scratch
        # CRITICAL: requires_grad=True so gradients flow
        self.time_embeddings = nn.Parameter(
            torch.zeros(vocab_size, time_dim),
            requires_grad=True  # SELF-LEARNING: gradients update this
        )

        # Statistics for analysis (not used in forward pass)
        self.register_buffer('usage_counts', torch.zeros(vocab_size))
        self.register_buffer('last_seen_step', torch.zeros(vocab_size))
        self.global_step = 0

        # Optional: Learnable update mechanism for inference
        # This learns HOW to update time, not what to update
        if learning_mode == 'hybrid':
            self.update_mlp = nn.Sequential(
                nn.Linear(time_dim + 1, time_dim),  # +1 for confidence score
                nn.Tanh(),
                nn.Linear(time_dim, time_dim)
            )

    def forward(self, token_ids, update_time=False, confidence_scores=None):
        """
        Get time embeddings for tokens.

        Args:
            token_ids: [batch_size, seq_len]
            update_time: Whether to update statistics (inference-time learning)
            confidence_scores: [batch_size, seq_len] optional confidence for updates

        Returns:
            time_emb: [batch_size, seq_len, time_dim]
        """
        # Get time embeddings - gradients flow through this automatically
        time_emb = self.time_embeddings[token_ids]

        # Optional: Inference-time updates (not used during training)
        if update_time and not self.training and self.learning_mode == 'hybrid':
            time_emb = self._apply_inference_update(token_ids, time_emb, confidence_scores)

        # Track statistics for analysis (doesn't affect gradients)
        if update_time:
            self._update_statistics(token_ids)

        return time_emb

    def _apply_inference_update(self, token_ids, time_emb, confidence_scores):
        """
        Optional: Update time during inference based on LEARNED patterns.
        This is NOT hardcoded - the update_mlp learns how to update.
        """
        with torch.no_grad():
            if confidence_scores is None:
                confidence_scores = torch.ones_like(token_ids, dtype=torch.float32)

            # Learnable update (model figures out what this means)
            conf_expanded = confidence_scores.unsqueeze(-1)
            update_input = torch.cat([time_emb, conf_expanded], dim=-1)
            delta = self.update_mlp(update_input)

            # Apply update to original embeddings (persistent)
            batch_size, seq_len = token_ids.shape
            for b in range(batch_size):
                for s in range(seq_len):
                    tid = token_ids[b, s].item()
                    self.time_embeddings.data[tid] += 0.01 * delta[b, s]

        return self.time_embeddings[token_ids]

    def _update_statistics(self, token_ids):
        """Track statistics for analysis (not used in model)"""
        with torch.no_grad():
            unique_tokens = token_ids.unique()
            for tid in unique_tokens:
                self.usage_counts[tid] += (token_ids == tid).sum()
                self.last_seen_step[tid] = self.global_step
            self.global_step += 1

    def get_statistics(self):
        """Get analysis statistics"""
        with torch.no_grad():
            time_magnitudes = torch.norm(self.time_embeddings, dim=1)

            return {
                'mean_time_magnitude': time_magnitudes.mean().item(),
                'max_time_magnitude': time_magnitudes.max().item(),
                'min_time_magnitude': time_magnitudes.min().item(),
                'std_time_magnitude': time_magnitudes.std().item(),
                'usage_counts': self.usage_counts.cpu().numpy(),
                'time_magnitudes': time_magnitudes.cpu().numpy(),
                'time_embeddings_sample': self.time_embeddings[:10].cpu().numpy(),
            }

    def reset_statistics(self):
        """Reset tracking statistics"""
        self.usage_counts.zero_()
        self.last_seen_step.zero_()
        self.global_step = 0


class Temporal_Tokenizer(nn.Module):
    """
    Manages [content | time] dual embeddings with self-learning time.
    """

    def __init__(self, vocab_size, content_dim, time_dim, learning_mode='gradient'):
        super().__init__()
        self.vocab_size = vocab_size
        self.content_dim = content_dim
        self.time_dim = time_dim
        self.total_dim = content_dim + time_dim

        # Content embeddings (standard)
        self.content_embeddings = nn.Embedding(vocab_size, content_dim)

        # Time embeddings (self-learning)
        self.time_embeddings = Temporal_TimeEmbeddings(vocab_size, time_dim, learning_mode)

        # Initialize content embeddings
        nn.init.normal_(self.content_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, update_time=False, confidence_scores=None):
        """
        Get [content | time] embeddings.

        During training: Both content and time learn through gradients
        During inference: Optional time updates based on usage
        """
        # Content embeddings
        content_emb = self.content_embeddings(token_ids)

        # Time embeddings (learns through gradients!)
        time_emb = self.time_embeddings(token_ids, update_time, confidence_scores)

        # Concatenate [content | time]
        embeddings = torch.cat([content_emb, time_emb], dim=-1)

        return embeddings

    def get_time_statistics(self):
        """Get time embedding statistics for analysis"""
        return self.time_embeddings.get_statistics()

    def analyze_learned_patterns(self):
        """
        Analyze what patterns the time embeddings discovered.
        This is for research analysis - shows what the model learned.
        """
        stats = self.get_time_statistics()
        time_matrix = self.time_embeddings.time_embeddings.detach().cpu()
        usage = torch.tensor(stats['usage_counts'])

        # Analyze each dimension
        analysis = {'dimensions': []}

        for dim in range(min(self.time_dim, 10)):  # Analyze first 10 dims
            dim_values = time_matrix[:, dim]

            # Correlation with frequency
            mask = usage > 0
            if mask.sum() > 10:
                freq_corr = torch.corrcoef(torch.stack([
                    dim_values[mask],
                    usage[mask].float()
                ]))[0, 1].item()
            else:
                freq_corr = 0.0

            analysis['dimensions'].append({
                'dim': dim,
                'mean': dim_values.mean().item(),
                'std': dim_values.std().item(),
                'freq_correlation': freq_corr,
                'range': (dim_values.min().item(), dim_values.max().item())
            })

        return analysis
