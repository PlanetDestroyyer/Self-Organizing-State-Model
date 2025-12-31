# Domain-Specific Generation Prompts for SOSM/Baseline Comparison

## Simple Wikipedia (Natural Language)
WIKI_PROMPTS = [
    "The solar system consists of",
    "Machine learning is a field of",
    "The Great Wall of China was built",
    "Photosynthesis is the process by which",
    "The internet was invented in",
    "Democracy is a form of government where",
    "Climate change refers to",
    "The human brain contains",
    "Einstein's theory of relativity states that",
    "Water is composed of",
    "The Renaissance was a period of",
    "Artificial intelligence can be defined as",
]

## Python Code
CODE_PROMPTS = [
    "def calculate_fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n    ",
    "class DataProcessor:\n    def __init__(self, data):\n        ",
    "import numpy as np\n\ndef matrix_multiply(a, b):\n    ",
    "# Binary search implementation\ndef binary_search(arr, target):\n    ",
    "from typing import List\n\ndef merge_sort(arr: List[int]) -> List[int]:\n    ",
    "class NeuralNetwork:\n    def __init__(self, layers):\n        self.layers = layers\n        ",
    "def read_csv(filename):\n    \"\"\"Read a CSV file and return data.\"\"\"\n    ",
    "# Implement a simple cache decorator\ndef cache(func):\n    ",
    "import torch\n\nclass Transformer(torch.nn.Module):\n    def __init__(self, d_model, nhead):\n        ",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ",
    "# Flask web application\nfrom flask import Flask\napp = Flask(__name__)\n\n@app.route('/')\ndef index():\n    ",
    "async def fetch_data(url):\n    \"\"\"Async function to fetch data from URL.\"\"\"\n    ",
]

## ArXiv Papers (Scientific Articles)
ARXIV_PROMPTS = [
    "Abstract: In this paper, we propose a novel approach to",
    "We present a comprehensive study of neural network architectures that",
    "This work introduces a new method for optimizing",
    "Recent advances in deep learning have shown that",
    "Our research investigates the relationship between",
    "We demonstrate that transformer-based models can",
    "This paper addresses the challenge of",
    "Experimental results indicate that our proposed method",
    "We analyze the performance of various machine learning algorithms for",
    "The main contribution of this work is",
    "In this study, we explore the efficacy of",
    "We propose a framework for understanding",
]
