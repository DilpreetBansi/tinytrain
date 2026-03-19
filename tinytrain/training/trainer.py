"""
Main Training Loop

Handles distributed training across different parallelism strategies.
Supports checkpointing, logging, and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple
import time
from tinytrain.distributed.comm import get_rank, is_distributed, barrier
from tinytrain.training.mixed_precision import GradScaler
from tinytrain.utils.logging import get_logger
from tinytrain.utils.metrics import MetricsTracker


logger = get_logger(__name__)


class Trainer:
    """
    Main training loop.

    Handles:
    - Forward/backward passes
    - Gradient synchronization for distributed training
    - Mixed precision training
    - Learning rate scheduling
    - Checkpointing
    - Metric tracking

    Args:
        model: Model to train
        optimizer: Optimizer
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        num_epochs: Number training epochs
        gradient_accumulation_steps: Gradient accumulation steps
        gradient_clip_norm: Max gradient norm for clipping
        enable_mixed_precision: Whether to use mixed precision
        enable_gradient_checkpoint: Whether to use gradient checkpointing
        checkpoint_dir: Directory to save checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        scheduler=None,
        device: torch.device = None,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float = 1.0,
        enable_mixed_precision: bool = False,
        enable_gradient_checkpoint: bool = False,
        checkpoint_dir: str = "./checkpoints",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_gradient_checkpoint = enable_gradient_checkpoint
        self.checkpoint_dir = checkpoint_dir

        # Move model to device
        self.model = self.model.to(self.device)

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=enable_mixed_precision)

        # Metrics
        self.metrics = MetricsTracker()

        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")

    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Dictionary with training results
        """
        logger.info(
            f"Starting training: {self.num_epochs} epochs, "
            f"device={self.device}, distributed={is_distributed()}"
        )

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Train one epoch
            epoch_loss = self._train_epoch(epoch)

            # Evaluate
            if self.eval_dataloader is not None:
                eval_loss = self._evaluate()
                logger.info(
                    f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}, Eval Loss: {eval_loss:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}")

        # Collect results
        results = {
            "final_loss": epoch_loss,
            "total_steps": self.global_step,
            "metrics": self.metrics.get_all(),
        }

        return results

    def _train_epoch(self, epoch: int) -> float:
        """Train one epoch."""
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (input_ids, labels) in enumerate(self.train_dataloader):
            # Move to device
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits, loss = self.model(
                input_ids=input_ids,
                labels=labels,
                return_loss=True,
            )

            # Scale loss if using mixed precision
            if self.enable_mixed_precision:
                scaled_loss = self.scaler.scale_loss(loss)
            else:
                scaled_loss = loss

            # Backward pass
            scaled_loss.backward()

            # Check for overflow
            if self.enable_mixed_precision:
                overflow = self.scaler.has_overflow(
                    [p.grad for p in self.model.parameters()]
                )
            else:
                overflow = False

            if not overflow:
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_norm,
                    )

                # Optimizer step
                self.optimizer.step()

                # Update loss scale
                if self.enable_mixed_precision:
                    self.scaler.step(self.optimizer, overflow=False)

                # Learning rate step
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            # Zero gradients
            self.optimizer.zero_grad()

            # Metrics
            epoch_loss += loss.item()
            num_batches += 1

            # Logging
            if (batch_idx + 1) % 10 == 0 and get_rank() == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(
                    f"  Batch {batch_idx + 1} - Loss: {avg_loss:.4f}, "
                    f"Global Step: {self.global_step}"
                )

        return epoch_loss / num_batches

    def _evaluate(self) -> float:
        """Evaluate on validation set."""
        if self.eval_dataloader is None:
            return 0.0

        self.model.eval()
        eval_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for input_ids, labels in self.eval_dataloader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                _, loss = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    return_loss=True,
                )

                eval_loss += loss.item()
                num_batches += 1

        return eval_loss / num_batches

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model checkpoint."""
        if get_rank() != 0:
            return

        checkpoint = {
            "epoch": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics.get_all(),
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
