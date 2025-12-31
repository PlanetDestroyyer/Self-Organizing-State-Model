# Generation Examples Summary

## Overview
Both `train_sosm_only.py` and `train_baseline_only.py` now include comprehensive text generation after training on each dataset. This provides qualitative comparison across three domains.

## Generation Configuration

### Prompts per Domain
- **Simple Wikipedia (Natural Language)**: 12 prompts covering diverse topics
- **Python Code**: 12 code prompts (functions, classes, algorithms)
- **ArXiv Papers (Scientific Articles)**: 12 abstract prompts

### Generation Parameters
- **Max Length**: 200 tokens per generation
- **Examples per Domain**: 12 (can be adjusted via `num_examples` parameter)
- **Total Examples per Model**: 36 (12 per domain × 3 domains)
- **Method**: Greedy decoding (argmax) for reproducibility

## Domain-Specific Prompts

### Wikipedia Text Examples
```
- "The solar system consists of"
- "Machine learning is a field of"
- "The Great Wall of China was built"
- "Photosynthesis is the process by which"
- "The internet was invented in"
- "Democracy is a form of government where"
- "Climate change refers to"
- "The human brain contains"
- "Einstein's theory of relativity states that"
- "Water is composed of"
- "The Renaissance was a period of"
- "Artificial intelligence can be defined as"
```

### Python Code Examples
```python
- "def calculate_fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n    "
- "class DataProcessor:\n    def __init__(self, data):\n        "
- "import numpy as np\n\ndef matrix_multiply(a, b):\n    "
- "# Binary search implementation\ndef binary_search(arr, target):\n    "
- "from typing import List\n\ndef merge_sort(arr: List[int]) -> List[int]:\n    "
- "class NeuralNetwork:\n    def __init__(self, layers):\n        self.layers = layers\n        "
- "def read_csv(filename):\n    \"\"\"Read a CSV file and return data.\"\"\"\n    "
- "# Implement a simple cache decorator\ndef cache(func):\n    "
- "import torch\n\nclass Transformer(torch.nn.Module):\n    def __init__(self, d_model, nhead):\n        "
- "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    "
- "# Flask web application\nfrom flask import Flask\napp = Flask(__name__)\n\n@app.route('/')\ndef index():\n    "
- "async def fetch_data(url):\n    \"\"\"Async function to fetch data from URL.\"\"\"\n    "
```

### ArXiv Scientific Articles
```
- "Abstract: In this paper, we propose a novel approach to"
- "We present a comprehensive study of neural network architectures that"
- "This work introduces a new method for optimizing"
- "Recent advances in deep learning have shown that"
- "Our research investigates the relationship between"
- "We demonstrate that transformer-based models can"
- "This paper addresses the challenge of"
- "Experimental results indicate that our proposed method"
- "We analyze the performance of various machine learning algorithms for"
- "The main contribution of this work is"
- "In this study, we explore the efficacy of"
- "We propose a framework for understanding"
```

## Output Format

### Console Output
Each generation is printed to console with:
- Example number
- Prompt (truncated to 100 chars if longer)
- Generated text (truncated to 500 chars if longer)
- Separator line

### JSON Output
Results are saved to `results/sosm_results.json` and `results/baseline_results.json` with structure:
```json
{
  "datasets": {
    "simple_wiki": [
      {
        "epoch": 1,
        "train_loss": ...,
        "test_loss": ...,
        "perplexity": ...
      },
      ...
      {
        "generation_examples": [
          {
            "prompt": "...",
            "generated": "...",
            "length": ...
          }
        ]
      }
    ]
  }
}
```

## Usage

### Training with Generation
```bash
# SOSM (15 epochs per dataset + generation)
python train_sosm_only.py

# Baseline (30 epochs per dataset + generation)
python train_baseline_only.py
```

### Expected Behavior
1. Train on Simple Wikipedia (15/30 epochs)
2. Generate 12 Wikipedia text examples
3. Train on Python Code (15/30 epochs)
4. Generate 12 Python code examples
5. Train on ArXiv Papers (15/30 epochs)
6. Generate 12 scientific article examples
7. Save all results including generations to JSON

## Evaluation Criteria

### Qualitative Comparison
- **Coherence**: Does the text make sense?
- **Domain Knowledge**: Does it demonstrate understanding of the domain?
- **Syntax**: For code, is it syntactically correct?
- **Completeness**: Does it complete the prompt meaningfully?
- **Creativity**: Does it show diverse responses?

### Expected SOSM Advantages
- More coherent text due to graph-constrained attention
- Better semantic understanding from MU blocks
- More consistent with domain conventions
- Less repetitiveness (baseline may overfit and repeat)
- Better long-range dependencies via shortcut edges

## Time Estimate
- Generation adds ~2-3 minutes per dataset (36 examples @ ~5 examples/min)
- Total generation time: ~6-9 minutes per model
- Negligible compared to training time (~10 hours)
