"""
Pipeline Parallelism

GPipe-style pipeline parallelism for very large models.
Splits model layers across stages (GPUs) with micro-batching.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class PipelineStage(nn.Module):
    """
    A stage in a pipeline parallel model.

    Contains a subset of model layers that runs on a specific device.

    Args:
        layers: List of model layers
        stage_id: Stage identifier
        num_stages: Total number of stages
    """

    def __init__(
        self,
        layers: List[nn.Module],
        stage_id: int,
        num_stages: int,
    ) -> None:
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute layers in this stage."""
        return self.layers(x)


class PipelineParallel(nn.Module):
    """
    Pipeline parallel model executor.

    Splits a model across multiple stages and executes them
    with micro-batching to keep the pipeline busy.

    Uses GPipe-style scheduling:
    1. Forward pass: Run through all stages with micro-batches
    2. Backward pass: Run in reverse order

    Args:
        stages: List of PipelineStage modules
        num_microbatches: Number of microbatches to create
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        num_microbatches: int = 4,
    ) -> None:
        super().__init__()

        self.stages = nn.ModuleList(stages)
        self.num_microbatches = num_microbatches
        self.num_stages = len(stages)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with pipeline schedule.

        Args:
            x: Input tensor, shape (batch_size, ...)
            labels: Optional labels for loss computation

        Returns:
            Tuple of (logits, loss)
        """
        batch_size = x.size(0)
        microbatch_size = max(1, batch_size // self.num_microbatches)

        # Split batch into microbatches
        microbatches = []
        for i in range(0, batch_size, microbatch_size):
            end = min(i + microbatch_size, batch_size)
            microbatches.append(x[i:end])

        # Forward pass through all stages with microbatches
        activations = [[] for _ in range(self.num_stages)]
        outputs = []

        for mb_idx, mb in enumerate(microbatches):
            # Forward through pipeline
            x_stage = mb
            for stage_idx, stage in enumerate(self.stages):
                x_stage = stage(x_stage)
                if stage_idx < self.num_stages - 1:
                    activations[stage_idx].append(x_stage.detach())

            outputs.append(x_stage)

        # Concatenate outputs
        logits = torch.cat(outputs, dim=0)
        loss = None

        if labels is not None:
            # Compute loss (simplified: just use last stage output)
            loss = self._compute_loss(logits, labels)

        return logits, loss

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute language modeling loss."""
        import torch.nn.functional as F

        # Flatten for cross-entropy
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
