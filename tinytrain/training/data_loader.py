"""
Data Loading and Tokenization

Efficient distributed data loading with automatic tokenization.
"""

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import List, Optional, Tuple
import numpy as np


class TextDataset(Dataset):
    """
    Simple text dataset that tokenizes on-the-fly.

    Args:
        data: List of text examples
        tokenizer_name: Name of tiktoken tokenizer ("gpt2", "cl100k_base", etc.)
        max_seq_len: Maximum sequence length (truncate longer sequences)
    """

    def __init__(
        self,
        data: List[str],
        tokenizer_name: str = "gpt2",
        max_seq_len: int = 1024,
    ) -> None:
        self.data = data
        self.max_seq_len = max_seq_len

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except KeyError:
            # Fallback to gpt2 if not found
            self.tokenizer = tiktoken.get_encoding("gpt2")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single example.

        Returns:
            Tuple of (input_ids, labels)
            Both shape (max_seq_len,)
        """
        text = self.data[idx]

        # Tokenize
        tokens = self.tokenizer.encode(text)

        # Truncate or pad to max_seq_len
        if len(tokens) >= self.max_seq_len:
            tokens = tokens[: self.max_seq_len]
        else:
            # Pad with end-of-text token (tokenizer.eot_token = 50256)
            tokens = tokens + [50256] * (self.max_seq_len - len(tokens))

        # Input and target (language modeling: predict next token)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, labels


class DataLoaderFactory:
    """Factory for creating distributed data loaders."""

    @staticmethod
    def create_dataloader(
        data: List[str],
        batch_size: int,
        max_seq_len: int = 1024,
        tokenizer_name: str = "gpt2",
        num_workers: int = 0,
        distributed: bool = False,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Create a data loader with optional distributed sampling.

        Args:
            data: List of text examples
            batch_size: Batch size
            max_seq_len: Maximum sequence length
            tokenizer_name: Tokenizer to use
            num_workers: Number of data loading workers
            distributed: Whether to use DistributedSampler
            shuffle: Whether to shuffle data

        Returns:
            DataLoader
        """
        dataset = TextDataset(
            data=data,
            tokenizer_name=tokenizer_name,
            max_seq_len=max_seq_len,
        )

        sampler = None
        if distributed:
            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                drop_last=True,
            )
            shuffle = False  # Don't shuffle when using sampler

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        return dataloader


def create_dummy_data(num_examples: int = 1000) -> List[str]:
    """
    Create dummy text data for testing.

    Args:
        num_examples: Number of examples to create

    Returns:
        List of text examples
    """
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language.",
        "Machine learning enables computers to learn from data.",
        "Transformers have revolutionized natural language processing.",
        "Attention is all you need for sequence modeling.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning has achieved remarkable performance on many tasks.",
        "Gradient descent is used to optimize neural networks.",
        "Backpropagation is a key algorithm for training deep models.",
        "Transfer learning helps leverage pre-trained models.",
    ]

    data = []
    for i in range(num_examples):
        # Cycle through sentences and concatenate
        text = " ".join([sentences[j % len(sentences)] for j in range((i % 5) + 1)])
        data.append(text)

    return data
