"""
Data loader for K-1 Self-Learning System.

Supports WikiText-2 and synthetic datasets.
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List, Dict, Union
from collections import Counter
import urllib.request
import zipfile


class K1_DataLoader:
    """
    Unified data loader for language modeling.

    Supports:
    - WikiText-2 (auto-download)
    - Synthetic text
    - Custom text files
    """

    def __init__(
        self,
        dataset_name: str = 'wikitext',
        data_dir: str = 'data',
        vocab_size: int = 10000,
        seq_length: int = 64,
        shared_vocab: 'DataLoader' = None  # For continual learning: share vocab across datasets
    ):
        """
        Initialize data loader.

        Args:
            dataset_name: 'wikitext', 'synthetic', or path to custom file
            data_dir: Directory for data storage
            vocab_size: Maximum vocabulary size
            seq_length: Sequence length for training
            shared_vocab: Optional DataLoader to copy vocabulary from (for continual learning)
        """
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.vocab_size = vocab_size
        self.seq_length = seq_length

        # Vocabulary
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab: List[str] = []
        self.token_to_id = self.word_to_idx  # Alias for compatibility
        
        # If shared_vocab provided, copy its vocabulary
        self._shared_vocab = shared_vocab

        # Data splits
        self.train_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self.val_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self.test_data: List[Tuple[np.ndarray, np.ndarray]] = []

        # Embeddings
        self._embeddings = None

        # Automatically load data
        self.load()

    def load(self):
        """Load and preprocess dataset."""
        print(f"Loading {self.dataset_name} dataset...")

        if self.dataset_name == 'wikitext':
            text = self._load_wikitext()
        elif self.dataset_name == 'code_python':
            text = self._load_code_python()
        elif self.dataset_name == 'scientific':
            text = self._load_scientific()
        elif self.dataset_name == 'synthetic':
            text = self._generate_synthetic()
        else:
            # Assume it's a file path
            text = self._load_custom(self.dataset_name)

        # Build vocabulary (or use shared vocab for continual learning)
        if self._shared_vocab is not None:
            print("Using shared vocabulary from previous dataset...")
            self.vocab = self._shared_vocab.vocab.copy()
            self.word_to_idx = self._shared_vocab.word_to_idx.copy()
            self.idx_to_word = self._shared_vocab.idx_to_word.copy()
            self.vocab_size = len(self.vocab)
            self.token_to_id = self.word_to_idx
        else:
            print("Building vocabulary...")
            self._build_vocabulary(text)

        # Convert to sequences
        print("Creating sequences...")
        sequences = self._text_to_sequences(text)

        # Split data
        n_train = int(len(sequences) * 0.9)
        n_val = int(len(sequences) * 0.05)

        self.train_data = sequences[:n_train]
        self.val_data = sequences[n_train:n_train + n_val]
        self.test_data = sequences[n_train + n_val:]

        # OPTIMIZATION: Convert to PyTorch tensors and move to GPU immediately
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print("Converting data to GPU tensors...")
            # Stack all sequences into single tensors on GPU
            train_x = torch.from_numpy(np.stack([s[0] for s in self.train_data])).long().to(device)
            train_y = torch.from_numpy(np.stack([s[1] for s in self.train_data])).long().to(device)
            self.train_data = (train_x, train_y)

            val_x = torch.from_numpy(np.stack([s[0] for s in self.val_data])).long().to(device)
            val_y = torch.from_numpy(np.stack([s[1] for s in self.val_data])).long().to(device)
            self.val_data = (val_x, val_y)

            test_x = torch.from_numpy(np.stack([s[0] for s in self.test_data])).long().to(device)
            test_y = torch.from_numpy(np.stack([s[1] for s in self.test_data])).long().to(device)
            self.test_data = (test_x, test_y)

            print(f"✓ All data on GPU! Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        print(f"\nDataset loaded:")
        print(f"  Train: {train_x.shape[0] if device.type == 'cuda' else len(self.train_data):,} sequences")
        print(f"  Val: {val_x.shape[0] if device.type == 'cuda' else len(self.val_data):,} sequences")
        print(f"  Test: {test_x.shape[0] if device.type == 'cuda' else len(self.test_data):,} sequences")
        print(f"  Vocab: {len(self.vocab):,} words")
        print(f"  Storage: {'GPU (fast!)' if device.type == 'cuda' else 'CPU'}")

    def _ensure_datasets_library(self):
        """Ensure datasets library is installed."""
        try:
            import datasets
            return datasets
        except ImportError:
            print("Installing 'datasets' library...")
            import subprocess
            subprocess.check_call(['pip', 'install', '-q', 'datasets'])
            import datasets
            return datasets

    def _load_code_python(self) -> str:
        """Load Python code dataset from HuggingFace."""
        data_dir = self.data_dir / 'code_python'
        file_path = data_dir / 'train.txt'

        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        print("Downloading Python Code dataset (nampdn-ai/tiny-codes)...")
        try:
            datasets = self._ensure_datasets_library()
            # Use nampdn-ai/tiny-codes - simple dataset without loading script
            dataset = datasets.load_dataset(
                'nampdn-ai/tiny-codes',
                split='train',
                streaming=True
            )

            print("Fetching 5,000 Python code examples...")
            texts = []
            for i, item in enumerate(dataset):
                if i >= 5000:
                    break
                # Get code from response field
                code = item.get('response', '') or item.get('code', '') or str(item)
                if len(code) > 100:
                    texts.append(code[:2000])

            if len(texts) < 100:
                raise ValueError("Not enough code examples downloaded")

            text = '\n\n'.join(texts)

            data_dir.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"Successfully downloaded {len(texts)} Python code examples!")
            return text

        except Exception as e:
            print(f"Download failed: {e}")
            print("Using synthetic python code...")
            return self._generate_synthetic_code()

    def _load_scientific(self) -> str:
        """Load Scientific dataset (ArXiv)."""
        data_dir = self.data_dir / 'scientific'
        file_path = data_dir / 'train.txt'
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        print("Downloading Scientific dataset (ArXiv)...")
        try:
            datasets = self._ensure_datasets_library()
            # Use ccdv/arxiv-summarization without trust_remote_code
            dataset = datasets.load_dataset(
                'ccdv/arxiv-summarization', 
                split='train', 
                streaming=True
            )
            
            print("Fetching 5,000 scientific abstracts...")
            texts = []
            for i, item in enumerate(dataset):
                if i >= 5000: break
                abstract = item.get('abstract', '')
                if len(abstract) > 100:
                    texts.append(abstract[:2000])
            
            text = '\n\n'.join(texts)
            
            data_dir.mkdir(exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
                
            return text
            
        except Exception as e:
            print(f"Download failed: {e}")
            print("Using synthetic scientific text...")
            return self._generate_synthetic_scientific()

    def _generate_synthetic_code(self) -> str:
        """Generate realistic synthetic Python code if download fails."""
        import random

        # More realistic Python code templates
        templates = [
            # Function definitions
            """def {func}({args}):
    \"\"\"Calculate {description}.\"\"\"
    result = {expr}
    return result
""",
            # Class definitions
            """class {class_name}:
    \"\"\"A class to handle {description}.\"\"\"

    def __init__(self, {args}):
        self.{attr} = {args}
        self.{attr2} = []

    def {method}(self, value):
        self.{attr2}.append(value)
        return len(self.{attr2})
""",
            # Data processing
            """def process_{data_type}(data):
    \"\"\"Process {data_type} data.\"\"\"
    results = []
    for item in data:
        if item.{attr} > 0:
            results.append(item.{method}())
    return results
""",
            # Error handling
            """try:
    result = {func}({args})
    print(f"Success: {{result}}")
except {exception} as e:
    print(f"Error: {{e}}")
    result = None
""",
            # List comprehensions
            """data = [{expr} for {var} in range({num})]
filtered = [x for x in data if x > {threshold}]
result = sum(filtered)
""",
            # Dictionary operations
            """config = {{
    '{key1}': {value1},
    '{key2}': '{value2}',
    '{key3}': [{expr} for i in range(5)]
}}
""",
            # API-like function
            """def fetch_{resource}(id):
    \"\"\"Fetch {resource} by ID.\"\"\"
    url = f"api/{{id}}/{resource}"
    response = request.get(url)
    return response.json()
""",
        ]

        # Vocabulary for code generation
        funcs = ['calculate', 'process', 'handle', 'parse', 'validate', 'transform', 'filter', 'sort']
        classes = ['DataProcessor', 'ConfigManager', 'ModelTrainer', 'FileHandler', 'APIClient', 'Cache', 'Parser']
        attrs = ['value', 'data', 'config', 'result', 'items', 'count', 'status']
        methods = ['get_value', 'process', 'update', 'validate', 'compute', 'transform']
        data_types = ['user', 'file', 'config', 'model', 'request', 'response']
        args_list = ['x', 'data', 'value', 'config', 'items']
        exceptions = ['ValueError', 'TypeError', 'KeyError', 'RuntimeError']

        text = []
        for _ in range(5000):
            template = random.choice(templates)
            code = template.format(
                func=random.choice(funcs),
                class_name=random.choice(classes),
                attr=random.choice(attrs),
                attr2=random.choice(attrs),
                method=random.choice(methods),
                data_type=random.choice(data_types),
                args=random.choice(args_list),
                description=f"{random.choice(data_types)} {random.choice(['processing', 'handling', 'validation'])}",
                expr=f"{random.choice(args_list)} * {random.randint(1, 10)}",
                exception=random.choice(exceptions),
                var='i',
                num=random.randint(10, 100),
                threshold=random.randint(10, 50),
                key1=random.choice(['host', 'port', 'timeout', 'debug']),
                key2=random.choice(['name', 'version', 'env', 'mode']),
                key3=random.choice(['options', 'features', 'plugins']),
                value1=random.randint(1000, 9999),
                value2=random.choice(['production', 'development', 'test']),
                resource=random.choice(data_types)
            )
            text.append(code)

        return "\n\n".join(text)

    def _generate_synthetic_scientific(self) -> str:
        """Generate synthetic scientific text if download fails."""
        templates = [
            "We propose a novel method for {topic} using {method}.",
            "The results show significant improvement in {metric}.",
            "Previous studies on {topic} have failed to address {issue}.",
            "Our algorithm optimizes {metric} by 20% compared to baseline."
        ]
        topics = ["neural networks", "quantum computing", "gene editing", "climate models"]
        methods = ["deep learning", "bayesian inference", "CRISPR", "simulation"]

        text = []
        for _ in range(5000):
            import random
            t = random.choice(templates)
            text.append(t.format(topic=random.choice(topics),
                               method=random.choice(methods),
                               metric="accuracy",
                               issue="scalability"))
        return "\n".join(text)

    def _load_wikitext(self) -> str:
        """Load WikiText-2 dataset."""
        wikitext_dir = self.data_dir / 'wikitext-2'

        if not wikitext_dir.exists():
            self._download_wikitext()

        # Load training text
        train_path = wikitext_dir / 'wiki.train.tokens'

        if train_path.exists():
            with open(train_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print("WikiText-2 not found, generating synthetic data...")
            return self._generate_synthetic()

    def _download_wikitext(self):
        """Download WikiText-2 dataset using HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing 'datasets' library...")
            import subprocess
            subprocess.check_call(['pip', 'install', '-q', 'datasets'])
            from datasets import load_dataset

        wikitext_dir = self.data_dir / 'wikitext-2'

        print("Downloading WikiText-2 from HuggingFace...")

        try:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=str(self.data_dir))

            # Create directory and save text files
            wikitext_dir.mkdir(exist_ok=True)

            print("Saving dataset files...")
            with open(wikitext_dir / 'wiki.train.tokens', 'w', encoding='utf-8') as f:
                f.write('\n'.join(dataset['train']['text']))

            with open(wikitext_dir / 'wiki.valid.tokens', 'w', encoding='utf-8') as f:
                f.write('\n'.join(dataset['validation']['text']))

            with open(wikitext_dir / 'wiki.test.tokens', 'w', encoding='utf-8') as f:
                f.write('\n'.join(dataset['test']['text']))

            print("WikiText-2 downloaded successfully")

        except Exception as e:
            print(f"Download failed: {e}")
            print("Using synthetic data instead")

    def _generate_synthetic(self) -> str:
        """Generate synthetic training text."""
        print("Generating synthetic text data...")

        # Templates for generating text
        templates = [
            "the {adj} {noun} {verb} {adv} in the {place}",
            "a {adj} {noun} and a {adj} {noun} {verb} together",
            "{noun} is {adj} and {adj}",
            "the {noun} {verb} while the {noun} {verb}",
            "in the {place} there was a {adj} {noun}",
            "the {adj} {noun} could {verb} very {adv}",
        ]

        words = {
            'adj': ['big', 'small', 'happy', 'sad', 'red', 'blue', 'fast', 'slow',
                   'bright', 'dark', 'old', 'new', 'good', 'bad', 'hot', 'cold'],
            'noun': ['cat', 'dog', 'bird', 'tree', 'house', 'car', 'book', 'sun',
                    'moon', 'star', 'river', 'mountain', 'city', 'garden', 'road', 'sky'],
            'verb': ['runs', 'jumps', 'sleeps', 'eats', 'plays', 'sings', 'dances', 'flies',
                    'swims', 'walks', 'talks', 'thinks', 'dreams', 'works', 'grows', 'moves'],
            'adv': ['quickly', 'slowly', 'happily', 'sadly', 'loudly', 'quietly',
                   'carefully', 'easily', 'hardly', 'nearly', 'always', 'never'],
            'place': ['forest', 'city', 'garden', 'house', 'park', 'school',
                     'library', 'kitchen', 'bedroom', 'street', 'beach', 'mountain']
        }

        sentences = []
        for _ in range(50000):  # Generate 50k sentences
            template = np.random.choice(templates)

            sentence = template
            for category, options in words.items():
                while f'{{{category}}}' in sentence:
                    sentence = sentence.replace(f'{{{category}}}', np.random.choice(options), 1)

            sentences.append(sentence)

        return ' . '.join(sentences)

    def _load_custom(self, path: str) -> str:
        """Load custom text file."""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _build_vocabulary(self, text: str):
        """Build vocabulary from text."""
        words = text.lower().split()
        word_counts = Counter(words)

        # Most common words (minus special tokens)
        most_common = word_counts.most_common(self.vocab_size - 4)

        # Special tokens
        self.vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + [w for w, _ in most_common]

        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}

    def _text_to_sequences(self, text: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Convert text to training sequences."""
        words = text.lower().split()

        # Convert to indices
        indices = [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in words]

        # Create sequences
        sequences = []
        stride = self.seq_length // 2  # 50% overlap

        for i in range(0, len(indices) - self.seq_length - 1, stride):
            x = np.array(indices[i:i + self.seq_length])
            y = np.array(indices[i + 1:i + self.seq_length + 1])
            sequences.append((x, y))

        return sequences

    def get_batch(self, split: str = 'train', batch_size: int = 32,
                  return_tensors: str = 'np') -> Union[Tuple[np.ndarray, np.ndarray],
                                                         Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a random batch - ULTRA FAST if data is on GPU!

        Args:
            split: 'train', 'val', or 'test'
            batch_size: Batch size
            return_tensors: 'np' for numpy, 'pt' for PyTorch tensors

        Returns:
            (batch_x, batch_y) - shapes (batch_size, seq_length)
        """
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        else:
            data = self.test_data

        # Check if data is already GPU tensors (tuple of 2 tensors)
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], torch.Tensor):
            # DATA IS ON GPU! Super fast path
            data_x, data_y = data
            n_samples = data_x.shape[0]
            actual_batch_size = min(batch_size, n_samples)

            # Random indexing directly on GPU
            indices = torch.randint(0, n_samples, (actual_batch_size,), device=data_x.device)

            # Index directly on GPU - NO CPU-GPU TRANSFER!
            batch_x = data_x[indices]
            batch_y = data_y[indices]

            return batch_x, batch_y

        # Fallback for CPU data (legacy path)
        if not data:
            raise ValueError(f"No data for split '{split}'")

        actual_batch_size = min(batch_size, len(data))
        indices = np.random.randint(0, len(data), size=actual_batch_size)

        batch_x = np.stack([data[i][0] for i in indices])
        batch_y = np.stack([data[i][1] for i in indices])

        if return_tensors == 'pt':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_x = torch.from_numpy(batch_x).long().to(device)
            batch_y = torch.from_numpy(batch_y).long().to(device)

        return batch_x, batch_y

    def get_embeddings(self, embedding_dim: int = 128) -> np.ndarray:
        """
        Get word embeddings (random initialization).

        Args:
            embedding_dim: Embedding dimension

        Returns:
            Embedding matrix (vocab_size, embedding_dim)
        """
        if self._embeddings is None or self._embeddings.shape[1] != embedding_dim:
            self._embeddings = np.random.randn(len(self.vocab), embedding_dim) * 0.02

        return self._embeddings

    def decode(self, indices: Union[List[int], np.ndarray]) -> str:
        """Decode indices to text."""
        words = [self.idx_to_word.get(i, '<UNK>') for i in indices]
        return ' '.join(words)
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to indices."""
        words = text.lower().split()
        return [self.word_to_idx.get(word, self.word_to_idx.get('<UNK>', 0)) for word in words]

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

