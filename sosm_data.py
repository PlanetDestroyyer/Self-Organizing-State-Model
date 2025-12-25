"""
Shared Data Module for Self-Organizing State Model

Loads WikiText, Code (Python), and Scientific (ArXiv) datasets.
Used by both test_base.py and test_sosm.py.
"""

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Tuple, List, Optional
import os


# Shared vocabulary (built once, used by all instances)
_SHARED_VOCAB = None


class MultiDomainDataset(Dataset):
    """
    Multi-domain dataset combining WikiText, Code, and Scientific text.
    Uses a SHARED vocabulary across all domains for consistent tokenization.
    """
    
    DOMAINS = ['wikitext', 'code', 'scientific']
    
    def __init__(
        self,
        domains: List[str] = None,
        vocab_size: int = 50257,
        seq_length: int = 64,
        split: str = 'train',
        max_samples_per_domain: int = None
    ):
        """
        Initialize multi-domain dataset.
        
        Args:
            domains: List of domains to load (default: all)
            vocab_size: Vocabulary size
            seq_length: Sequence length
            split: 'train' or 'test'
            max_samples_per_domain: Maximum samples per domain
        """
        self.domains = domains or self.DOMAINS
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.split = split
        self.max_samples = max_samples_per_domain
        
        # Build shared vocabulary (only once)
        self._build_shared_vocab()
        
        # Load datasets
        self.data = []
        self.domain_labels = []
        
        for domain in self.domains:
            samples = self._load_domain(domain, split)
            self.data.extend(samples)
            self.domain_labels.extend([domain] * len(samples))
        
        print(f"Loaded {len(self.data)} samples from {len(self.domains)} domains")
    
    def _load_domain(self, domain: str, split: str) -> List[torch.Tensor]:
        """Load a specific domain."""
        try:
            from datasets import load_dataset
        except ImportError:
            print("Warning: datasets not installed, using synthetic data")
            return self._generate_synthetic(domain, 1000)
        
        samples = []
        
        try:
            if domain == 'wikitext':
                ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
                text = '\n'.join([t for t in ds['text'] if t.strip()])
                samples = self._tokenize_text(text)
                
            elif domain == 'code':
                # Use code dataset - use default config
                try:
                    ds = load_dataset('bigcode/the-stack-smol', split='train', streaming=True)
                    text = ''
                    for i, item in enumerate(ds):
                        if i >= 500:  # Limit samples
                            break
                        content = item.get('content', '')
                        if content:
                            text += content + '\n'
                    samples = self._tokenize_text(text)
                except Exception:
                    # Fallback to codeparrot/github-code-clean
                    ds = load_dataset('codeparrot/github-code-clean', 
                                     streaming=True, split='train',
                                     languages=['Python'])
                    text = ''
                    for i, item in enumerate(ds):
                        if i >= 500:
                            break
                        text += item.get('code', '') + '\n'
                    samples = self._tokenize_text(text)
                
            elif domain == 'scientific':
                # Use ArXiv dataset
                ds = load_dataset('ccdv/arxiv-summarization', split='train', streaming=True)
                text = ''
                for i, item in enumerate(ds):
                    if i >= 500:
                        break
                    text += item.get('abstract', '') + '\n'
                samples = self._tokenize_text(text)
                
        except Exception as e:
            print(f"Warning: Could not load {domain}: {e}")
            samples = self._generate_synthetic(domain, 1000)
        
        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]
        
        print(f"  {domain}: {len(samples)} samples")
        return samples
    
    def _build_shared_vocab(self):
        """Initialize BPE tokenizer (GPT-2 style)."""
        global _SHARED_VOCAB
        
        if _SHARED_VOCAB is None:
            try:
                from transformers import GPT2TokenizerFast
                tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
                tokenizer.pad_token = tokenizer.eos_token
                _SHARED_VOCAB = {'tokenizer': tokenizer, 'type': 'bpe'}
                print(f"  Using GPT-2 BPE tokenizer: {tokenizer.vocab_size} tokens")
            except Exception as e:
                print(f"  Warning: Could not load GPT-2 tokenizer ({e}), using fallback")
                # Fallback to simple vocab
                chars = " \t\n" + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "0123456789" + ".,;:!?'\"-()[]{}"
                char_to_idx = {c: i for i, c in enumerate(chars)}
                idx_to_char = {i: c for c, i in char_to_idx.items()}
                _SHARED_VOCAB = {'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char, 'type': 'char'}
                print(f"  Using fallback char vocab: {len(chars)} chars")
        
        self._vocab = _SHARED_VOCAB
    
    def _tokenize_text(self, text: str) -> List[torch.Tensor]:
        """Tokenize text using BPE or character-level tokenizer."""
        if self._vocab.get('type') == 'bpe':
            tokenizer = self._vocab['tokenizer']
            # Tokenize entire text
            encoding = tokenizer.encode(text, add_special_tokens=False)
            tokens = encoding
        else:
            # Fallback character-level
            char_to_idx = self._vocab['char_to_idx']
            tokens = [char_to_idx.get(c, 0) for c in text]
        
        # Create sequences
        samples = []
        for i in range(0, len(tokens) - self.seq_length, self.seq_length):
            seq = tokens[i:i + self.seq_length]
            if len(seq) == self.seq_length:
                samples.append(torch.tensor(seq, dtype=torch.long))
        
        return samples
    
    def get_vocab(self) -> Tuple[dict, dict]:
        """Get vocabulary info for saving with checkpoints."""
        if self._vocab.get('type') == 'bpe':
            tokenizer = self._vocab['tokenizer']
            return {'type': 'bpe', 'name': 'gpt2'}, {'vocab_size': tokenizer.vocab_size}
        else:
            return self._vocab['char_to_idx'], self._vocab['idx_to_char']
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if self._vocab.get('type') == 'bpe':
            tokenizer = self._vocab['tokenizer']
            return tokenizer.decode(token_ids)
        else:
            idx_to_char = self._vocab['idx_to_char']
            return ''.join(idx_to_char.get(t, '?') for t in token_ids)
    
    def _generate_synthetic(self, domain: str, count: int) -> List[torch.Tensor]:
        """Generate synthetic data if real data unavailable."""
        samples = []
        for _ in range(count):
            seq = torch.randint(0, self.vocab_size, (self.seq_length,))
            samples.append(seq)
        return samples
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get item.
        
        Returns:
            input_ids: [seq_length]
            labels: [seq_length] (shifted by 1)
            domain: domain name
        """
        tokens = self.data[idx]
        domain = self.domain_labels[idx]
        
        input_ids = tokens[:-1] if len(tokens) > 1 else tokens
        labels = tokens[1:] if len(tokens) > 1 else tokens
        
        # Ensure correct length
        if len(input_ids) < self.seq_length - 1:
            pad_len = self.seq_length - 1 - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.zeros(pad_len, dtype=torch.long)])
        
        return input_ids, labels, domain


def create_dataloaders(
    batch_size: int = 64,
    vocab_size: int = 50257,
    seq_length: int = 64,
    domains: List[str] = None
) -> Tuple[TorchDataLoader, TorchDataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        batch_size: Batch size (default: 64)
        vocab_size: Vocabulary size
        seq_length: Sequence length
        domains: Domains to include
        
    Returns:
        train_loader, test_loader
    """
    def collate_fn(batch):
        input_ids = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        domains = [b[2] for b in batch]
        return input_ids, labels, domains
    
    train_dataset = MultiDomainDataset(
        domains=domains,
        vocab_size=vocab_size,
        seq_length=seq_length,
        split='train'
    )
    
    test_dataset = MultiDomainDataset(
        domains=domains,
        vocab_size=vocab_size,
        seq_length=seq_length,
        split='test'
    )
    
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, test_loader


def load_simple_wikipedia(
    tokenizer,
    max_length: int = 512,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    split: str = 'train'
) -> TorchDataLoader:
    """
    Load Simple Wikipedia dataset for Phase 2.4.
    
    Args:
        tokenizer: HuggingFace tokenizer (GPT-2)
        max_length: Maximum sequence length
        batch_size: Batch size
        max_samples: Limit number of samples (None = all)
        split: 'train' or 'validation'
        
    Returns:
        DataLoader with Simple Wikipedia data
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required. Run: pip install datasets")
    
    print(f"Loading Simple Wikipedia ({split})...")
    
    # Load Simple Wikipedia
    # Note: Using wikimedia/wikipedia (Parquet-based) to avoid script loading errors
    # Config: 20231101.simple (November 2023 dump)
    try:
        dataset = load_dataset('wikimedia/wikipedia', '20231101.simple', split=split)
    except Exception:
        # Fallback to the generic config if specific date fails
        print("  Warning: Specific date config failed, trying generic '20231101.simple'")
        dataset = load_dataset('wikimedia/wikipedia', '20231101.simple', split=split)
    
    print(f"  Loaded {len(dataset)} articles")
    
    # Filter out short articles
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 100)
    
    print(f"  After filtering: {len(dataset)} articles")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
    
    # Process in batches
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['id', 'url', 'title', 'text']
    )
    
    # Set format
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # Limit samples if specified
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"  Limited to {len(dataset)} samples")
    
    print(f"✅ Simple Wikipedia loaded: {len(dataset)} sequences")
    
    # Create DataLoader
    # Enable shuffle for any train split (including slices like 'train[:90%]')
    do_shuffle = split.startswith('train')
    
    dataloader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=0
    )
    
    return dataloader


if __name__ == '__main__':
    # Test
    train_loader, test_loader = create_dataloaders(batch_size=64)
    for batch in train_loader:
        input_ids, labels, domains = batch
        print(f"Batch: {input_ids.shape}, Domains: {set(domains)}")
        break
