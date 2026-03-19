#!/usr/bin/env python3
"""
Prepare training data.

Downloads and tokenizes text data for training.
Uses tiny Shakespeare dataset by default (public domain).
"""

import os
import urllib.request
from pathlib import Path


def download_tiny_shakespeare(output_path: str = "data/tiny_shakespeare.txt") -> None:
    """
    Download tiny Shakespeare dataset.

    This is a small public domain dataset of Shakespeare's works.
    Perfect for testing and demo purposes.

    Args:
        output_path: Where to save the file
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return

    print(f"Downloading tiny Shakespeare to {output_path}...")

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded {output_path}")

        # Print stats
        with open(output_path, 'r') as f:
            text = f.read()
        print(f"File size: {len(text) / 1e6:.2f} MB")
        print(f"Number of lines: {len(text.splitlines())}")

    except Exception as e:
        print(f"Error downloading file: {e}")
        print("Using dummy data instead")
        create_dummy_data(output_path)


def create_dummy_data(output_path: str = "data/dummy_data.txt") -> None:
    """
    Create dummy training data.

    Args:
        output_path: Where to save the file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    text = """
    The machine learning revolution has transformed our world.
    Transformers have become the dominant architecture for natural language processing.
    Attention is all you need for sequence modeling tasks.
    Neural networks learn patterns from data through backpropagation.
    Deep learning enables computers to learn hierarchical representations.
    Large language models exhibit remarkable few-shot learning abilities.
    Distributed training allows us to scale model sizes dramatically.
    Efficient attention mechanisms reduce computational complexity.
    Gradient checkpointing trades compute for memory to enable larger models.
    Mixed precision training accelerates training while maintaining accuracy.
    Pipeline parallelism distributes model layers across multiple GPUs.
    Tensor parallelism shards weight matrices for even larger models.
    """ * 100  # Repeat to create larger file

    with open(output_path, 'w') as f:
        f.write(text)

    print(f"Created dummy data: {output_path}")


def main():
    """Download or create training data."""
    # Try to download tiny Shakespeare
    download_tiny_shakespeare()


if __name__ == "__main__":
    main()
