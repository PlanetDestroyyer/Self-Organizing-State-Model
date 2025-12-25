"""
Long Context & Factual Recall Test Suite

Tests what SOSM actually learned from WikiText-2:
1. Factual recall (WikiText-appropriate)
2. Long-context reasoning
3. Multi-hop inference
4. Semantic coherence

Usage:
    python test_long_context.py
"""

import torch
import sys
from transformers import GPT2Tokenizer
from state_core.pipeline import StateCorePipeline

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


# ============================================================================
# TEST SETS
# ============================================================================

# WikiText-2 likely has good coverage of these topics
FACTUAL_TESTS = [
    # US Presidents (WikiText has many political articles)
    {
        'context': 'Barack Obama was the president of',
        'expected_contains': ['united', 'states', 'america'],
        'category': 'Politics'
    },
    {
        'context': 'George Washington was the first president of the',
        'expected_contains': ['united', 'states'],
        'category': 'Politics'
    },
    
    # Technology (WikiText is rich in tech articles)
    {
        'context': 'The iPhone was made by',
        'expected_contains': ['apple', 'inc'],
        'category': 'Technology'
    },
    {
        'context': 'Microsoft was founded by',
        'expected_contains': ['bill', 'gates'],
        'category': 'Technology'
    },
    
    # Geography (common knowledge)
    {
        'context': 'New York is a city in',
        'expected_contains': ['new', 'york', 'state', 'united'],
        'category': 'Geography'
    },
    {
        'context': 'London is the capital of',
        'expected_contains': ['england', 'britain', 'kingdom'],
        'category': 'Geography'
    },
    
    # Entertainment (WikiText has film/book articles)
    {
        'context': 'Harry Potter is a series of',
        'expected_contains': ['books', 'novels', 'fantasy'],
        'category': 'Entertainment'
    },
    {
        'context': 'Star Wars was directed by',
        'expected_contains': ['george', 'lucas'],
        'category': 'Entertainment'
    },
    
    # Sports (WikiText has sports coverage)
    {
        'context': 'The Super Bowl is a championship game for',
        'expected_contains': ['football', 'nfl'],
        'category': 'Sports'
    },
    
    # Science (common facts)
    {
        'context': 'The Earth orbits around the',
        'expected_contains': ['sun', 'star'],
        'category': 'Science'
    },
]


LONG_CONTEXT_TESTS = [
    # Test 1: Multi-sentence reasoning
    {
        'context': """The company was founded in 1976 by Steve Jobs and Steve Wozniak. 
They started in a garage in California. The company's first product was a personal computer. 
Today, the company is known for""",
        'expected_topic': 'technology/products',
        'length': 'medium',
        'test': 'multi_sentence'
    },
    
    # Test 2: Narrative continuation
    {
        'context': """In the year 2000, the internet was becoming mainstream. 
People were buying computers for their homes. Email was replacing traditional mail. 
Companies were rushing to create websites. This period is often called""",
        'expected_topic': 'technology/era',
        'length': 'medium',
        'test': 'narrative'
    },
    
    # Test 3: Cause and effect
    {
        'context': """During World War II, many countries mobilized their economies for war production. 
Factories that once made consumer goods began producing weapons and ammunition. 
Women entered the workforce in large numbers to replace men who had gone to fight. 
As a result, the war economy""",
        'expected_topic': 'economic impact',
        'length': 'long',
        'test': 'reasoning'
    },
    
    # Test 4: Technical description
    {
        'context': """A computer processor executes instructions from memory. 
It retrieves data, performs calculations, and stores results. 
Modern processors can execute billions of instructions per second. 
The speed of a processor is measured in""",
        'expected_topic': 'measurement/units',
        'length': 'medium',
        'test': 'technical'
    },
    
    # Test 5: Very long context (test graph limits)
    {
        'context': """The Renaissance was a period of great cultural change and achievement in Europe. 
It began in Italy in the 14th century and spread throughout Europe over the next few centuries. 
Renaissance art was characterized by realism and attention to detail. Famous Renaissance artists include Leonardo da Vinci, Michelangelo, and Raphael. 
Da Vinci painted the Mona Lisa, one of the most famous paintings in the world. 
Michelangelo sculpted David and painted the Sistine Chapel ceiling. 
The Renaissance also saw advances in science, with figures like Galileo challenging traditional ideas about the universe. 
Literature flourished during this time, with writers like Shakespeare in England and Dante in Italy. 
The invention of the printing press by Gutenberg helped spread ideas more quickly. 
Overall, the Renaissance marked a transition from medieval to""",
        'expected_topic': 'modern/era',
        'length': 'very_long',
        'test': 'long_context'
    },
]


DISAMBIGUATION_TESTS = [
    # Test ambiguous words in context
    {
        'context1': 'The baseball player swung the bat and',
        'context2': 'The bat flew through the cave and',
        'word': 'bat',
        'should_differ': True
    },
    {
        'context1': 'The bank of the river was covered in',
        'context2': 'The bank on Main Street offers loans for',
        'word': 'bank',
        'should_differ': True
    },
    {
        'context1': 'In the spring, flowers begin to',
        'context2': 'The metal spring in the watch is used to',
        'word': 'spring',
        'should_differ': True
    },
]


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def load_model(checkpoint_path='sosm_trained.pt', device='cuda'):
    """Load trained SOSM model."""
    print(f"{BLUE}Loading model from {checkpoint_path}...{RESET}")
    
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    
    # Ensure graph is enabled
    if 'pipeline' in config and 'graph' in config['pipeline']:
        config['pipeline']['graph']['enabled'] = True
    
    pipeline = StateCorePipeline(config).to(device)
    pipeline.load_state_dict(checkpoint['model_state_dict'])
    pipeline.eval()
    
    print(f"{GREEN}✓ Model loaded{RESET}\n")
    return pipeline


def test_factual_recall(pipeline, tokenizer, tests, device='cuda', top_k=5):
    """Test factual recall on WikiText-appropriate queries."""
    print("=" * 70)
    print(f"{BOLD}FACTUAL RECALL TEST{RESET}")
    print("=" * 70)
    print(f"Testing {len(tests)} factual queries...\n")
    
    results = []
    
    for i, test in enumerate(tests, 1):
        context = test['context']
        expected = test['expected_contains']
        category = test['category']
        
        print(f"{BOLD}Test {i}/{len(tests)}: {category}{RESET}")
        print(f"Query: {BLUE}\"{context}\"{RESET}")
        
        # Tokenize
        tokens = tokenizer.encode(context, return_tensors='pt').to(device)
        
        # Get prediction
        with torch.no_grad():
            logits, _ = pipeline(tokens, return_state=True)
        
        # Get top-k predictions
        last_logits = logits[0, -1, :]
        top_k_ids = torch.topk(last_logits, top_k).indices
        top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_ids]
        
        # Check if any expected word appears in top-k
        top_k_lower = [t.lower().strip() for t in top_k_tokens]
        matches = []
        for exp in expected:
            if any(exp.lower() in tok for tok in top_k_lower):
                matches.append(exp)
        
        success = len(matches) > 0
        
        print(f"Top-{top_k} predictions: {top_k_tokens}")
        print(f"Expected (any of): {expected}")
        
        if success:
            print(f"{GREEN}✓ PASS{RESET} - Found: {matches}")
        else:
            print(f"{RED}✗ FAIL{RESET} - None of expected words in top-{top_k}")
        
        print()
        
        results.append({
            'category': category,
            'query': context,
            'expected': expected,
            'predicted': top_k_tokens,
            'success': success,
            'matches': matches
        })
    
    # Summary
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print("=" * 70)
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"Passed: {GREEN}{passed}/{total}{RESET} ({100*passed/total:.1f}%)")
    
    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'passed': 0, 'total': 0}
        categories[cat]['total'] += 1
        if r['success']:
            categories[cat]['passed'] += 1
    
    print("\nPer-category:")
    for cat, stats in categories.items():
        pct = 100 * stats['passed'] / stats['total']
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({pct:.0f}%)")
    
    print("=" * 70)
    print()
    
    return results


def test_long_context(pipeline, tokenizer, tests, device='cuda', gen_length=10):
    """Test long-context understanding."""
    print("=" * 70)
    print(f"{BOLD}LONG CONTEXT TEST{RESET}")
    print("=" * 70)
    print(f"Testing {len(tests)} long-context scenarios...\n")
    
    results = []
    
    for i, test in enumerate(tests, 1):
        context = test['context']
        expected_topic = test['expected_topic']
        test_type = test['test']
        length = test['length']
        
        print(f"{BOLD}Test {i}/{len(tests)}: {test_type.upper()} ({length}){RESET}")
        print(f"Context ({len(context.split())} words):")
        print(f"{BLUE}{context[:200]}...{RESET}" if len(context) > 200 else f"{BLUE}{context}{RESET}")
        print()
        
        # Tokenize
        tokens = tokenizer.encode(context, return_tensors='pt').to(device)
        seq_len = tokens.size(1)
        
        # Generate continuation
        with torch.no_grad():
            generated = tokens.clone()
            
            for _ in range(gen_length):
                logits, state = pipeline(generated, return_state=True)
                next_token = logits[0, -1, :].argmax()
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        # Decode
        continuation = tokenizer.decode(generated[0, seq_len:])
        
        print(f"Generated: {YELLOW}{continuation}{RESET}")
        print(f"Expected topic: {expected_topic}")
        
        # Check graph info
        if state.routing_state:
            num_edges = state.routing_state['num_edges']
            edge_types = state.routing_state['graph']['edge_types']
            print(f"Graph: {num_edges} edges ({edge_types})")
        
        print()
        
        results.append({
            'type': test_type,
            'length': length,
            'context_words': len(context.split()),
            'seq_len': seq_len,
            'continuation': continuation,
            'num_edges': num_edges if state.routing_state else 0
        })
    
    print("=" * 70)
    print()
    
    return results


def test_disambiguation(pipeline, tokenizer, tests, device='cuda'):
    """Test contextual disambiguation."""
    print("=" * 70)
    print(f"{BOLD}DISAMBIGUATION TEST{RESET}")
    print("=" * 70)
    print(f"Testing {len(tests)} ambiguous word pairs...\n")
    
    results = []
    
    for i, test in enumerate(tests, 1):
        word = test['word']
        ctx1 = test['context1']
        ctx2 = test['context2']
        
        print(f"{BOLD}Test {i}/{len(tests)}: '{word}'{RESET}")
        
        # Context 1
        tokens1 = tokenizer.encode(ctx1, return_tensors='pt').to(device)
        with torch.no_grad():
            logits1, state1 = pipeline(tokens1, return_state=True)
        pred1 = tokenizer.decode([logits1[0, -1, :].argmax()])
        
        # Context 2
        tokens2 = tokenizer.encode(ctx2, return_tensors='pt').to(device)
        with torch.no_grad():
            logits2, state2 = pipeline(tokens2, return_state=True)
        pred2 = tokenizer.decode([logits2[0, -1, :].argmax()])
        
        differ = pred1.strip() != pred2.strip()
        success = differ == test['should_differ']
        
        print(f"Context 1: {BLUE}{ctx1}{RESET}")
        print(f"Prediction 1: {YELLOW}{pred1}{RESET}")
        print(f"Context 2: {BLUE}{ctx2}{RESET}")
        print(f"Prediction 2: {YELLOW}{pred2}{RESET}")
        
        if success:
            print(f"{GREEN}✓ PASS{RESET} - Predictions {'differ' if differ else 'same'} as expected")
        else:
            print(f"{RED}✗ FAIL{RESET} - Predictions should {'differ' if test['should_differ'] else 'match'}")
        
        print()
        
        results.append({
            'word': word,
            'pred1': pred1,
            'pred2': pred2,
            'differ': differ,
            'success': success
        })
    
    # Summary
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print("=" * 70)
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"Passed: {GREEN}{passed}/{total}{RESET} ({100*passed/total:.1f}%)")
    print("=" * 70)
    print()
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    pipeline = load_model(device=device)
    
    # Run tests
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}COMPREHENSIVE TEST SUITE{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")
    
    # 1. Factual Recall
    factual_results = test_factual_recall(pipeline, tokenizer, FACTUAL_TESTS, device)
    
    # 2. Long Context
    long_context_results = test_long_context(pipeline, tokenizer, LONG_CONTEXT_TESTS, device)
    
    # 3. Disambiguation
    disambiguation_results = test_disambiguation(pipeline, tokenizer, DISAMBIGUATION_TESTS, device)
    
    # Overall summary
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}OVERALL RESULTS{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    
    factual_pass = sum(1 for r in factual_results if r['success'])
    disambig_pass = sum(1 for r in disambiguation_results if r['success'])
    
    print(f"\nFactual Recall: {factual_pass}/{len(factual_results)} passed")
    print(f"Disambiguation: {disambig_pass}/{len(disambiguation_results)} passed")
    print(f"Long Context: {len(long_context_results)} tests completed")
    
    print(f"\n{GREEN}Test suite complete!{RESET}\n")


if __name__ == '__main__':
    main()
