"""
Build Word-Level Vocabulary from Simple Wikipedia
==================================================

This script builds a word-level vocabulary tokenizer from the training data.
The tokenizer will then be used for SOSM training.
"""
import sys
from word_tokenizer import WordLevelTokenizer
from datasets import load_dataset


def main():
    print("="*70)
    print("BUILDING WORD-LEVEL VOCABULARY")
    print("="*70)
    
    # Load Simple Wikipedia
    print("\nLoading Simple Wikipedia dataset...")
    dataset = load_dataset('wikimedia/wikipedia', '20231101.simple', split='train')
    
    # Extract texts (limit to reduce memory)
    print(f"Extracting texts from {len(dataset)} articles...")
    texts = []
    for i, item in enumerate(dataset):
        if i >= 100000:  # Limit for vocab building
            break
        text = item['text']
        if len(text) > 200:  # Filter short articles
            texts.append(text)
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1} articles, collected {len(texts)} valid texts")
    
    print(f"\nTotal texts for vocabulary: {len(texts)}")
    
    # Build vocabulary
    tokenizer = WordLevelTokenizer()
    tokenizer.build_vocab(texts, vocab_size=100000)
    
    # Save tokenizer
    tokenizer.save('word_tokenizer_vocab.json')
    
    # Test
    print("\n" + "="*70)
    print("TESTING TOKENIZER")
    print("="*70)
    
    test_sentences = [
        "The solar system consists of planets.",
        "Machine learning is a field of artificial intelligence.",
        "The Great Wall of China was built over many centuries."
    ]
    
    for sentence in test_sentences:
        tokens = tokenizer.encode(sentence)
        decoded = tokenizer.decode(tokens)
        print(f"\nOriginal: {sentence}")
        print(f"Tokens ({len(tokens)}): {tokens[:10]}...")
        print(f"Decoded: {decoded}")
    
    print("\n✅ Vocabulary built successfully!")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Saved to: word_tokenizer_vocab.json")


if __name__ == "__main__":
    main()
