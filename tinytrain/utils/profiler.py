"""
Performance Profiling Utilities

Memory and time profiling for training.
"""

import torch
import time
from typing import Optional
from contextlib import contextmanager


class MemoryMonitor:
    """Monitor GPU memory usage during training."""

    def __init__(self) -> None:
        self.peak_allocated = 0
        self.peak_reserved = 0

    def reset(self) -> None:
        """Reset memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def update(self) -> None:
        """Update peak memory stats."""
        if torch.cuda.is_available():
            self.peak_allocated = torch.cuda.max_memory_allocated() / 1e9  # GB
            self.peak_reserved = torch.cuda.max_memory_reserved() / 1e9  # GB

    def get_stats(self) -> dict:
        """Get memory stats."""
        return {
            "peak_allocated_gb": self.peak_allocated,
            "peak_reserved_gb": self.peak_reserved,
        }


class TimingContext:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "block") -> None:
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed:.3f}s"


@contextmanager
def time_block(name: str = "block"):
    """Context manager to time a code block."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{name}: {elapsed:.3f}s")
