"""
AdamW Optimizer

Adam with decoupled weight decay (AdamW).
Weight decay is not applied to biases and LayerNorm parameters.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Tuple


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.

    Implements the algorithm from "Decoupled Weight Decay Regularization"
    (https://arxiv.org/abs/1711.05101).

    Unlike L2 regularization, weight decay is decoupled from the
    gradient-based update rule, leading to better generalization.

    Args:
        params: Model parameters
        lr: Learning rate
        betas: Coefficients for computing running averages (beta1, beta2)
        eps: Term added for numerical stability
        weight_decay: Weight decay coefficient (applied to all params)
        amsgrad: Whether to use AMSGrad variant
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get gradient
                grad = p.grad.data
                if grad.dtype in [torch.float16, torch.bfloat16]:
                    grad = grad.float()

                amsgrad = group["amsgrad"]

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay learning rate
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(
                        group["eps"]
                    )
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(
                        group["eps"]
                    )

                step_size = group["lr"] / bias_correction1

                # Weight decay (decoupled)
                # Do not decay biases and LayerNorm parameters
                if group["weight_decay"] != 0 and self._should_apply_weight_decay(
                    p
                ):
                    p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

                # Adam update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def _should_apply_weight_decay(self, param: torch.Tensor) -> bool:
        """
        Determine if weight decay should be applied to parameter.

        Weight decay is skipped for biases and LayerNorm/BatchNorm parameters.

        Args:
            param: Parameter tensor

        Returns:
            True if weight decay should be applied
        """
        # Skip 1D parameters (biases, layer norm weights/biases)
        if param.dim() == 1:
            return False

        return True
