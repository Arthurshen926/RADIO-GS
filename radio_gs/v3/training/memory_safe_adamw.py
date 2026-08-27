"""Exact AdamW equations with bounded transient memory for a joint D512."""

from __future__ import annotations

import math

import torch


class MemorySafeAdamW(torch.optim.Optimizer):
    """Apply AdamW in flat chunks without a parameter-sized denominator."""

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        chunk_elements: int = 1_048_576,
    ) -> None:
        if lr < 0 or eps < 0 or weight_decay < 0 or chunk_elements <= 0:
            raise ValueError("memory-safe AdamW hyperparameters differ")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("memory-safe AdamW betas differ")
        defaults = dict(
            lr=float(lr),
            betas=tuple(float(value) for value in betas),
            eps=float(eps),
            weight_decay=float(weight_decay),
            chunk_elements=int(chunk_elements),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("memory-safe AdamW requires dense gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = int(state["step"])
                first = state["exp_avg"].reshape(-1)
                second = state["exp_avg_sq"].reshape(-1)
                values = parameter.reshape(-1)
                gradients = gradient.reshape(-1)
                bias1 = 1.0 - beta1**step
                bias2_sqrt = math.sqrt(1.0 - beta2**step)
                step_size = group["lr"] / bias1
                chunk = int(group["chunk_elements"])
                for start in range(0, values.numel(), chunk):
                    stop = min(start + chunk, values.numel())
                    value = values[start:stop]
                    grad = gradients[start:stop]
                    mean = first[start:stop]
                    variance = second[start:stop]
                    if group["weight_decay"]:
                        value.mul_(1.0 - group["lr"] * group["weight_decay"])
                    mean.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denominator = variance.sqrt().div_(bias2_sqrt).add_(group["eps"])
                    value.addcdiv_(mean, denominator, value=-step_size)
        return loss


__all__ = ["MemorySafeAdamW"]
