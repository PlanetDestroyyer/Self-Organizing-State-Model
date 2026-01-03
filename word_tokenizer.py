"""
Word-Level Tokenizer for SOSM
==============================

Philosophy: SOSM's MU blocks are designed to extract semantic meaning.
BPE tokenization fragments words into meaningless subwords ("est", "kward"),
preventing semantic extraction.

This tokenizer operates at the word level, aligning with SOSM's architecture.

Usage:
    tokenizer = WordLevelTokenizer()
    tokenizer.build_vocab(texts, vocab_size=100000)
    tokens = tokenizer.encode("The solar system consists of planets")
    text = tokenizer.decode(tokens)
"""
import re
import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional


class WordLevelTokenizer:
    """Simple word-level tokenizer with special tokens."""
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    THINK_START = "<think>"
    THINK_END = "</think>"
    
    def __init__(self):
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Initialize with special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.THINK_START,  # CoT start
            self.THINK_END      # CoT end
        ]
        for token in special_tokens:
            self.word2id[token] = len(self.word2id)
            self.id2word[len(self.id2word)] = token
        
        self.vocab_size = len(self.word2id)
        self.pad_token_id = self.word2id[self.PAD_TOKEN]
        self.unk_token_id = self.word2id[self.UNK_TOKEN]
        self.bos_token_id = self.word2id[self.BOS_TOKEN]
        self.eos_token_id = self.word2id[self.EOS_TOKEN]
        self.think_start_id = self.word2id[self.THINK_START]
        self.think_end_id = self.word2id[self.THINK_END]
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Rules:
        - Lowercase everything
        - Split on whitespace and punctuation
        - Keep contractions together ("don't" → "don't")
        - Separate punctuation ("hello!" → ["hello", "!"])
        """
        # Lowercase
        text = text.lower()
        
        # Add spaces around punctuation (except apostrophes in contractions)
        text = re.sub(r"([.,!?;:(){}\[\]])", r" \1 ", text)
        
        # Split on whitespace and filter empty strings
        words = [w.strip() for w in text.split() if w.strip()]
        
        return words
    
    def build_vocab(self, texts: List[str], vocab_size: int = 100000):
        """
        Build vocabulary from text corpus.
        
        Args:
            texts: List of text documents
            vocab_size: Maximum vocabulary size (including special tokens)
        """
        print(f"Building word-level vocabulary from {len(texts)} documents...")
        
        # Count all words
        word_counts = Counter()
        for text in texts:
            words = self._tokenize_text(text)
            word_counts.update(words)
        
        print(f"  Found {len(word_counts)} unique words")
        
        # Keep most common words (accounting for special tokens already added)
        max_words = vocab_size - len(self.word2id)
        most_common = word_counts.most_common(max_words)
        
        # Add to vocabulary
        for word, count in most_common:
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)
                self.id2word[len(self.id2word)] = word
        
        self.vocab_size = len(self.word2id)
        
        print(f"  Final vocabulary size: {self.vocab_size}")
        print(f"  Coverage: {sum(c for w, c in most_common) / sum(word_counts.values()):.1%}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            List of token IDs
        """
        words = self._tokenize_text(text)
        
        # Convert to IDs
        token_ids = [
            self.word2id.get(word, self.unk_token_id)
            for word in words
        ]
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        
        words = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            words.append(self.id2word.get(token_id, self.UNK_TOKEN))
        
        # Join words with spaces
        # Smart punctuation spacing: no space before punctuation
        text = " ".join(words)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        
        return text
    
    def save(self, filepath: str):
        """Save vocabulary to file."""
        vocab_data = {
            'word2id': self.word2id,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        print(f"Saved tokenizer to {filepath}")
    
    def load(self, filepath: str):
        """Load vocabulary from file."""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        self.word2id = vocab_data['word2id']
        self.vocab_size = vocab_data['vocab_size']
        
        # Rebuild id2word
        self.id2word = {int(v): k for k, v in self.word2id.items()}
        
        # Refresh special token IDs
        self.pad_token_id = self.word2id[self.PAD_TOKEN]
        self.unk_token_id = self.word2id[self.UNK_TOKEN]
        self.bos_token_id = self.word2id[self.BOS_TOKEN]
        self.eos_token_id = self.word2id[self.EOS_TOKEN]
        self.think_start_id = self.word2id.get(self.THINK_START, self.unk_token_id)
        self.think_end_id = self.word2id.get(self.THINK_END, self.unk_token_id)
        
        print(f"Loaded tokenizer from {filepath} (vocab_size={self.vocab_size})")


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = WordLevelTokenizer()
    
    # Sample texts
    texts = [
        "The solar system consists of planets.",
        "Machine learning is a field of artificial intelligence.",
        "The Great Wall of China was built over many centuries."
    ]
    
    # Build vocab
    tokenizer.build_vocab(texts, vocab_size=1000)
    
    # Test encoding/decoding
    test_text = "The solar system consists of planets."
    print(f"\nOriginal: {test_text}")
    
    tokens = tokenizer.encode(test_text)
    print(f"Tokens: {tokens}")
    
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    
    # Test unknown word
    unk_text = "The zxqwerty consists of planets."
    tokens = tokenizer.encode(unk_text)
    decoded = tokenizer.decode(tokens)
    print(f"\nWith UNK: {unk_text} → {decoded}")
