"""
Multi-Dataset Loader for SOSM Comparison Experiment

Loads 3 datasets for rigorous comparison:
1. Simple Wikipedia (natural language)
2. Python Code (structured)
3. ArXiv Papers (scientific)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from typing import Dict, Tuple, List
import random


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input (all but last) and labels (all but first)
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # Pad if needed
        pad_len = self.max_length - 1 - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.eos_token_id] * pad_len
            labels = labels + [-100] * pad_len  # -100 is ignore index
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def load_simple_wikipedia(split: str = 'train', max_samples: int = 50000) -> List[str]:
    """Load Simple Wikipedia dataset using sosm_data approach"""
    print(f"Loading Simple Wikipedia ({split})...")
    
    # Use same approach as sosm_data.py (which works!)
    dataset = load_dataset('wikipedia', '20220301.simple', split=split, trust_remote_code=True)
    
    # Sample and filter
    if len(dataset) > max_samples:
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = dataset.select(indices)
    
    # Extract text and filter short articles
    texts = []
    for item in dataset:
        text = item['text']
        if len(text) > 200:  # Minimum length
            texts.append(text)
    
    print(f"  Loaded {len(texts)} articles")
    return texts


def load_python_code(max_samples: int = 50000) -> List[str]:
    """Load Python code from The Stack"""
    print("Loading Python code...")
    
    # Load Python subset of the-stack-smol
    dataset = load_dataset(
        'bigcode/the-stack-smol',
        data_dir='data/python',
        split='train',
        streaming=True,
        trust_remote_code=True
    )
    
    # Collect samples
    texts = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        if len(item['content']) > 100:
            texts.append(item['content'])
    
    print(f"  Loaded {len(texts)} code files")
    return texts


def load_arxiv_papers(max_samples: int = 50000) -> List[str]:
    """Load ArXiv scientific papers"""
    print("Loading ArXiv papers...")
    
    # Load ArXiv dataset
    dataset = load_dataset('ccdv/arxiv-summarization', split='train', trust_remote_code=True)
    
    # Sample
    if len(dataset) > max_samples:
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = dataset.select(indices)
    
    # Extract abstracts (more manageable than full papers)
    texts = [item['abstract'] for item in dataset if len(item.get('abstract', '')) > 100]
    
    print(f"  Loaded {len(texts)} papers")
    return texts


def create_multi_dataset_loaders(
    tokenizer,
    batch_size: int = 64,
    max_samples_per_dataset: int = 50000,
    max_length: int = 256
) -> Dict[str, Tuple[DataLoader, DataLoader]]:
    """
    Create train/test loaders for all 3 datasets
    
    Returns:
        dict with keys 'simple_wiki', 'code', 'arxiv'
        each value is (train_loader, test_loader)
    """
    loaders = {}
    
    # Simple Wikipedia
    texts = load_simple_wikipedia(max_samples=max_samples_per_dataset)
    split = int(len(texts) * 0.9)
    train_dataset = TextDataset(texts[:split], tokenizer, max_length)
    test_dataset = TextDataset(texts[split:], tokenizer, max_length)
    loaders['simple_wiki'] = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    )
    
    # Python Code
    texts = load_python_code(max_samples=max_samples_per_dataset)
    split = int(len(texts) * 0.9)
    train_dataset = TextDataset(texts[:split], tokenizer, max_length)
    test_dataset = TextDataset(texts[split:], tokenizer, max_length)
    loaders['code'] = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    )
    
    # ArXiv Papers
    texts = load_arxiv_papers(max_samples=max_samples_per_dataset)
    split = int(len(texts) * 0.9)
    train_dataset = TextDataset(texts[:split], tokenizer, max_length)
    test_dataset = TextDataset(texts[split:], tokenizer, max_length)
    loaders['arxiv'] = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    )
    
    return loaders


if __name__ == "__main__":
    # Test loading
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    loaders = create_multi_dataset_loaders(tokenizer, batch_size=32, max_samples_per_dataset=1000)
    
    for name, (train_loader, test_loader) in loaders.items():
        print(f"\n{name}:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test one batch
        batch = next(iter(train_loader))
        print(f"  Batch shape: {batch['input_ids'].shape}")
        print(f"  ✓ Loader working!")
