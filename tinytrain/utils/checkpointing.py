"""
Model and Optimizer State Management

Save and load checkpoints for distributed training.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os


class CheckpointManager:
    """Manage model and optimizer checkpoints."""

    def __init__(self, checkpoint_dir: str = "./checkpoints") -> None:
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        name: str = "checkpoint",
    ) -> str:
        """
        Save checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            step: Current step
            metrics: Optional metrics to save
            name: Checkpoint name prefix

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
        }

        path = os.path.join(
            self.checkpoint_dir,
            f"{name}-epoch{epoch}-step{step}.pt",
        )

        torch.save(checkpoint, path)
        return path

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            checkpoint_path: Path to checkpoint

        Returns:
            Dictionary with epoch, step, and metrics
        """
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return {
            "epoch": checkpoint["epoch"],
            "step": checkpoint["step"],
            "metrics": checkpoint.get("metrics", {}),
        }

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find latest checkpoint in directory."""
        files = [
            f
            for f in os.listdir(self.checkpoint_dir)
            if f.endswith(".pt")
        ]

        if not files:
            return None

        # Sort by modification time
        files.sort(
            key=lambda f: os.path.getmtime(os.path.join(self.checkpoint_dir, f)),
            reverse=True,
        )

        return os.path.join(self.checkpoint_dir, files[0])
