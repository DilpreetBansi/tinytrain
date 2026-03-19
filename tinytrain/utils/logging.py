"""
Distributed-aware logging.

Only rank 0 prints to avoid duplicate messages.
"""

import logging
import sys
from tinytrain.distributed.comm import get_rank


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger that only prints from rank 0.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Only rank 0 should log
        if get_rank() == 0:
            # Console handler
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        else:
            # Other ranks: disable logging
            logger.setLevel(logging.CRITICAL)

    return logger
