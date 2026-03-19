"""
TinyTrain: Distributed LLM Training Framework

A production-quality framework for distributed training of GPT models from scratch.
Supports data parallelism, tensor parallelism, and pipeline parallelism.
"""

__version__ = "0.1.0"
__author__ = "TinyTrain Contributors"

from tinytrain.model.config import GPTConfig
from tinytrain.model.gpt import GPT

__all__ = [
    "GPTConfig",
    "GPT",
    "__version__",
    "__author__",
]
