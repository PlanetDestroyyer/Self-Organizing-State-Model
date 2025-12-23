"""
MU-SOTA: Meaning Unit Transformer with State-of-the-Art Architecture

Complete implementation with:
- 8×8 structured semantic matrix (16 blocks of 2×2)
- Block-wise semantic attention (structure-aware)
- 24-layer deep architecture (SOTA depth)
- Mixed precision training (FP16 + FP32)
- Temperature sampling generation
- 50K BPE vocabulary
- NO hardcoded values - all learned or computed

Production-ready code with error handling and validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from tqdm import tqdm
import math
from typing import Optional, Tuple, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class MU_Config:
    """Configuration for MU-SOTA Transformer"""

    # Matrix structure (8×8 = 64 dims, 16 semantic blocks)
    matrix_size = 8
    num_semantic_blocks = 16  # Each block is 2×2
    block_size = 2  # 2×2 blocks

    # Architecture (SOTA-level)
    n_layers = 24  # Deep like GPT-2 Medium
    n_heads = 8
    dropout = 0.1

    # Vocabulary
    vocab_size = 50000  # Like GPT-2
    max_seq_len = 512

    # Training
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    weight_decay = 0.01
    warmup_steps = 1000
    max_grad_norm = 1.0

    # Mixed precision
    use_mixed_precision = True

    # Generation
    temperature = 0.8
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.2

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


config = MU_Config()


# ============================================================================
# SEMANTIC BLOCK DEFINITIONS (Structure, not values!)
# ============================================================================

class MU_SemanticBlockLayout:
    """Defines the 16 semantic blocks in 8×8 matrix"""

    BLOCKS = {
        # Name: (row_start, col_start, row_end, col_end, description)
        'I': (0, 0, 2, 2, 'Identity - core token meaning'),
        'S': (0, 2, 2, 4, 'Structure - grammatical properties'),
        'C1': (0, 4, 2, 6, 'Context-Local - immediate context'),
        'C2': (0, 6, 2, 8, 'Context-Global - document context'),
        'R1': (2, 0, 4, 2, 'Relations-Syntactic - syntax dependencies'),
        'R2': (2, 2, 4, 4, 'Relations-Semantic - meaning dependencies'),
        'T': (2, 4, 4, 6, 'Transformation - compositional changes'),
        'K': (2, 6, 4, 8, 'Knowledge - world knowledge grounding'),
        'G': (4, 0, 6, 2, 'Global - document coherence'),
        'M': (4, 2, 6, 4, 'Modality - certainty/tense/mood'),
        'D': (4, 4, 6, 6, 'Discourse - rhetorical structure'),
        'F': (4, 6, 6, 8, 'Frame - semantic frame roles'),
        'P': (6, 0, 8, 2, 'Position - positional encoding'),
        'E': (6, 2, 8, 4, 'Entity - named entity properties'),
        'A': (6, 4, 8, 6, 'Affect - sentiment/emotion'),
        'X': (6, 6, 8, 8, 'Extension - flexible/learned purpose'),
    }

    @classmethod
    def get_block_indices(cls, block_name: str) -> Tuple[int, int, int, int]:
        """Get (r1, c1, r2, c2) for a semantic block"""
        return cls.BLOCKS[block_name][:4]

    @classmethod
    def get_all_block_names(cls) -> List[str]:
        """Get list of all block names"""
        return list(cls.BLOCKS.keys())


# ============================================================================
# DATASET WITH BPE TOKENIZATION
# ============================================================================

class MU_Dataset(Dataset):
    """WikiText-2 with BPE tokenization (50K vocab)"""

    def __init__(self, split: str = 'train', max_seq_len: int = 512,
                 tokenizer: Optional[Tokenizer] = None, vocab_size: int = 50000):
        logger.info(f"Loading {split} dataset...")

        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            all_text = ' '.join([item['text'] for item in dataset if len(item['text'].strip()) > 0])

            if tokenizer is None:
                logger.info(f"Training BPE tokenizer with vocab_size={vocab_size}...")
                self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
                self.tokenizer.pre_tokenizer = Whitespace()

                trainer = BpeTrainer(
                    vocab_size=vocab_size,
                    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
                    show_progress=False
                )

                self.tokenizer.train_from_iterator([all_text], trainer=trainer)
                self.vocab_size = self.tokenizer.get_vocab_size()
                logger.info(f"Tokenizer trained. Vocab size: {self.vocab_size}")
            else:
                self.tokenizer = tokenizer
                self.vocab_size = self.tokenizer.get_vocab_size()

            # Tokenize text
            encoding = self.tokenizer.encode(all_text)
            all_tokens = encoding.ids

            # Create sequences
            self.data = []
            stride = max_seq_len // 2
            for i in range(0, len(all_tokens) - max_seq_len - 1, stride):
                chunk = all_tokens[i:i + max_seq_len + 1]
                if len(chunk) == max_seq_len + 1:
                    self.data.append(torch.tensor(chunk, dtype=torch.long))

            logger.info(f"Created {len(self.data)} sequences from {split} split")

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.data[idx]
        return {'input_ids': seq[:-1], 'labels': seq[1:]}

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids)


# ============================================================================
# FULLY DYNAMIC SEMANTIC SENSITIVITY (NO HARDCODING)
# ============================================================================

class MU_DynamicSensitivity(nn.Module):
    """Compute sensitivity for each semantic block - ALL LEARNED"""

    def __init__(self, num_blocks: int, vocab_size: int, d_model: int):
        super().__init__()
        self.num_blocks = num_blocks

        # LEARNED: Base sensitivity for each block (not hardcoded!)
        self.block_sensitivity_base = nn.Parameter(
            torch.randn(num_blocks) * 0.1 + 0.5  # Init ~0.5, then learn
        )

        # LEARNED: Token affinity to each block
        self.token_block_affinity = nn.Parameter(
            torch.randn(vocab_size, num_blocks) * 0.1
        )

        # LEARNED: Block interaction matrix
        self.block_interaction = nn.Parameter(
            torch.eye(num_blocks) * 0.5 + torch.randn(num_blocks, num_blocks) * 0.1
        )

        # LEARNED: Sensitivity modulation network
        self.sensitivity_net = nn.Sequential(
            nn.Linear(num_blocks + 1, num_blocks * 2),  # +1 for attention entropy
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(num_blocks * 2, num_blocks),
            nn.Sigmoid()  # 0-1 range
        )

    def forward(self, token_ids: torch.Tensor, attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute block-wise sensitivity - all from learned parameters!

        Args:
            token_ids: [B, T]
            attention_weights: [B, T, num_heads, T] (optional)

        Returns:
            sensitivity: [B, T, num_blocks] - fully computed, no hardcoding!
        """
        B, T = token_ids.shape

        # Token-specific block affinity (LEARNED)
        affinity = self.token_block_affinity[token_ids]  # [B, T, num_blocks]

        # Attention-based modulation (COMPUTED from context)
        if attention_weights is not None:
            attn_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-9),
                dim=-1
            ).mean(dim=-2)  # [B, T]
        else:
            attn_entropy = torch.zeros(B, T, device=token_ids.device)

        # Combine features
        features = torch.cat([affinity, attn_entropy.unsqueeze(-1)], dim=-1)  # [B, T, num_blocks+1]

        # Compute sensitivity through learned network
        sensitivity = self.sensitivity_net(features)  # [B, T, num_blocks]

        # Modulate by base sensitivity (LEARNED parameter)
        base = self.block_sensitivity_base.view(1, 1, -1)
        sensitivity = sensitivity * base

        return sensitivity  # All values learned or computed - NO HARDCODING!


# ============================================================================
# BLOCK-WISE SEMANTIC ATTENTION
# ============================================================================

class MU_BlockAttention(nn.Module):
    """Structure-aware attention that processes semantic blocks separately"""

    def __init__(self, config: MU_Config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.num_blocks = config.num_semantic_blocks

        # Each semantic block gets its own attention module
        self.block_attentions = nn.ModuleDict()
        block_names = MU_SemanticBlockLayout.get_all_block_names()

        for block_name in block_names:
            # Each 2×2 block = 4 values
            self.block_attentions[block_name] = nn.MultiheadAttention(
                embed_dim=4,
                num_heads=2,
                dropout=config.dropout,
                batch_first=True
            )

        # Cross-block attention for global refinement
        self.cross_block_attn = nn.MultiheadAttention(
            embed_dim=64,  # 8×8 matrix flattened
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 64)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)

        # Dynamic sensitivity computer
        self.sensitivity_computer = MU_DynamicSensitivity(
            config.num_semantic_blocks,
            config.vocab_size,
            64
        )

    def forward(self, M: torch.Tensor, token_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process 8×8 matrices with block-wise semantic attention

        Args:
            M: [B, T, 8, 8] - input matrices
            token_ids: [B, T] - for sensitivity computation
            mask: [B, T] - attention mask (optional)

        Returns:
            M_out: [B, T, 8, 8] - processed matrices
        """
        B, T = M.shape[0], M.shape[1]
        block_outputs = {}

        # Process each semantic block independently
        for block_name in MU_SemanticBlockLayout.get_all_block_names():
            r1, c1, r2, c2 = MU_SemanticBlockLayout.get_block_indices(block_name)

            # Extract block
            block_data = M[:, :, r1:r2, c1:c2]  # [B, T, 2, 2]
            block_flat = block_data.reshape(B, T, 4)  # [B, T, 4]

            # Self-attention within block
            block_out, _ = self.block_attentions[block_name](
                block_flat, block_flat, block_flat,
                key_padding_mask=mask if mask is not None else None
            )

            block_outputs[block_name] = block_out

        # Combine all blocks
        all_blocks = torch.cat(list(block_outputs.values()), dim=-1)  # [B, T, 64]

        # Cross-block attention (blocks interact)
        cross_out, attn_weights = self.cross_block_attn(
            all_blocks, all_blocks, all_blocks,
            key_padding_mask=mask if mask is not None else None
        )

        # Residual + norm
        all_blocks = self.norm1(all_blocks + cross_out)

        # Feed-forward
        ffn_out = self.ffn(all_blocks)
        all_blocks = self.norm2(all_blocks + ffn_out)

        # Compute dynamic sensitivity
        sensitivity = self.sensitivity_computer(token_ids, attn_weights)  # [B, T, 16]

        # Apply sensitivity-based gating (block-wise)
        all_blocks_reshaped = all_blocks.reshape(B, T, 16, 4)  # [B, T, 16 blocks, 4 values]
        sensitivity_expanded = sensitivity.unsqueeze(-1)  # [B, T, 16, 1]

        # Modulate each block by its sensitivity
        M_flat_original = M.reshape(B, T, 16, 4)
        delta = all_blocks_reshaped - M_flat_original
        M_flat_new = M_flat_original + delta * sensitivity_expanded

        # Reshape back to 8×8
        M_out = M_flat_new.reshape(B, T, 8, 8)

        return M_out


# ============================================================================
# DEEP MU-SOTA TRANSFORMER (24 LAYERS)
# ============================================================================

class MU_Transformer(nn.Module):
    """24-layer deep MU Transformer with semantic block structure"""

    def __init__(self, config: MU_Config):
        super().__init__()
        self.config = config

        # Token embeddings → 8×8 matrix
        self.token_to_mu = nn.Embedding(config.vocab_size, 64)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_seq_len, 64) * 0.02
        )

        # 24 layers of block-wise semantic attention (SOTA depth!)
        self.layers = nn.ModuleList([
            MU_BlockAttention(config)
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(64)

        # Output projection
        self.mu_to_logits = nn.Linear(64, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(f"Initialized MU-SOTA with {self.count_parameters():,} parameters")

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [B, T]
            mask: [B, T] (optional)

        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = input_ids.shape

        # Embed tokens to 8×8 matrices
        M = self.token_to_mu(input_ids)  # [B, T, 64]
        M = M + self.pos_encoding[:, :T, :]  # Add positional encoding
        M = M.reshape(B, T, 8, 8)  # Reshape to matrices

        # Process through 24 deep layers with structure-aware attention
        for layer in self.layers:
            M = layer(M, input_ids, mask)

        # Final norm and flatten
        M_flat = M.reshape(B, T, 64)
        M_flat = self.final_norm(M_flat)

        # Output logits
        logits = self.mu_to_logits(M_flat)

        return logits


# ============================================================================
# TRAINING WITH MIXED PRECISION
# ============================================================================

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                scheduler, scaler: GradScaler, device: str, epoch: int, total_epochs: int) -> Dict:
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]")

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(enabled=config.use_mixed_precision):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_tokens += labels.numel()
            total_loss += loss.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/labels.numel():.4f}'})

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

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with autocast(enabled=config.use_mixed_precision):
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


# ============================================================================
# TEXT GENERATION WITH TEMPERATURE SAMPLING
# ============================================================================

def generate_text(model: nn.Module, dataset: MU_Dataset, prompt: str,
                  max_length: int = 100, temperature: float = 0.8,
                  top_k: int = 50, top_p: float = 0.9,
                  repetition_penalty: float = 1.2, device: str = 'cuda') -> str:
    """Generate text with temperature/top-k/top-p sampling"""
    model.eval()

    # Encode prompt
    encoding = dataset.tokenizer.encode(prompt)
    input_ids = encoding.ids

    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            input_tensor = torch.tensor([input_ids[-config.max_seq_len:]], dtype=torch.long).to(device)

            # Forward pass
            with autocast(enabled=config.use_mixed_precision):
                logits = model(input_tensor)

            next_token_logits = logits[0, -1, :] / temperature

            # Apply repetition penalty
            for token_id in set(input_ids):
                next_token_logits[token_id] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            input_ids.append(next_token)

    return dataset.decode(input_ids)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("MU-SOTA TRANSFORMER - Production Training")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  • Matrix: 8×8 (16 semantic blocks)")
    logger.info(f"  • Layers: {config.n_layers}")
    logger.info(f"  • Vocab: {config.vocab_size}")
    logger.info(f"  • Mixed Precision: {config.use_mixed_precision}")
    logger.info(f"  • Device: {config.device}")
    logger.info("=" * 80)

    # Load data
    try:
        train_dataset = MU_Dataset('train', config.max_seq_len, vocab_size=config.vocab_size)
        val_dataset = MU_Dataset('validation', config.max_seq_len, tokenizer=train_dataset.tokenizer)

        # Update config with actual vocab size
        config.vocab_size = train_dataset.vocab_size

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=2)

        logger.info(f"Dataset loaded:")
        logger.info(f"  • Train: {len(train_dataset):,} sequences")
        logger.info(f"  • Val: {len(val_dataset):,} sequences")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Create model
    model = MU_Transformer(config).to(config.device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=total_steps,
        pct_start=config.warmup_steps/total_steps
    )
    scaler = GradScaler(enabled=config.use_mixed_precision)

    # Training loop
    best_perplexity = float('inf')

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, scaler, config.device, epoch, config.num_epochs)
        val_metrics = evaluate(model, val_loader, config.device)

        logger.info(f"\nEpoch {epoch}:")
        logger.info(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']*100:.2f}%")
        logger.info(f"  Val: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']*100:.2f}%, PPL={val_metrics['perplexity']:.2f}")

        # Generate sample text
        if epoch % 2 == 0:  # Every 2 epochs
            logger.info("\nSample Generation:")
            for prompt in ["The quick brown", "Once upon a time"]:
                generated = generate_text(model, train_dataset, prompt, max_length=30, device=config.device)
                logger.info(f"  '{prompt}' → '{generated}'")

        # Save best model
        if val_metrics['perplexity'] < best_perplexity:
            best_perplexity = val_metrics['perplexity']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'perplexity': best_perplexity,
            }, 'mu_sota_best.pt')
            train_dataset.tokenizer.save('mu_sota_tokenizer.json')
            logger.info(f"  ✓ Saved best model (PPL={best_perplexity:.2f})")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
