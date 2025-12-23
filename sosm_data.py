"""
Shared Data Module for Self-Organizing State Model

Loads WikiText, Code (Python), and Scientific (ArXiv) datasets.
Used by both test_base.py and test_sosm.py.
"""

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Tuple, List, Optional
import os


class MultiDomainDataset(Dataset):
    """
    Multi-domain dataset combining WikiText, Code, and Scientific text.
    """
    
    DOMAINS = ['wikitext', 'code', 'scientific']
    
    def __init__(
        self,
        domains: List[str] = None,
        vocab_size: int = 50000,
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
    
    def _tokenize_text(self, text: str) -> List[torch.Tensor]:
        """Simple character-level tokenization."""
        # Build vocab from text (simple approach)
        chars = sorted(set(text))
        char_to_idx = {c: i % self.vocab_size for i, c in enumerate(chars)}
        
        # Convert to token IDs
        tokens = [char_to_idx.get(c, 0) for c in text]
        
        # Create sequences
        samples = []
        for i in range(0, len(tokens) - self.seq_length, self.seq_length):
            seq = tokens[i:i + self.seq_length]
            if len(seq) == self.seq_length:
                samples.append(torch.tensor(seq, dtype=torch.long))
        
        return samples
    
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
    vocab_size: int = 50000,
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


if __name__ == '__main__':
    # Test
    train_loader, test_loader = create_dataloaders(batch_size=64)
    for batch in train_loader:
        input_ids, labels, domains = batch
        print(f"Batch: {input_ids.shape}, Domains: {set(domains)}")
        break
