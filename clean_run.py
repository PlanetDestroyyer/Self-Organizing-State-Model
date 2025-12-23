#!/usr/bin/env python3
"""
Clean Run Script - Clears pycache before running tests.
"""

import os
import shutil
import subprocess
import sys

def clear_pycache():
    """Remove all __pycache__ directories."""
    count = 0
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d == '__pycache__':
                path = os.path.join(root, d)
                shutil.rmtree(path)
                count += 1
    print(f"Cleared {count} __pycache__ directories")

if __name__ == '__main__':
    # Clear cache first
    clear_pycache()
    
    # Run the requested script
    args = sys.argv[1:] if len(sys.argv) > 1 else ['test_sosm.py', '--stage', '3', '--epochs', '3', '--batch-size', '64']
    
    print(f"Running: python {' '.join(args)}")
    print("=" * 70)
    
    subprocess.run([sys.executable] + args)
