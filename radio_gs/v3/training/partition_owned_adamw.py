"""AdamW with column-owned updates for one structured D512 parameter."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class PartitionOwnedAdamW:
    """Keep optimizer state and weight decay inside one owned column slice."""

    def __init__(
        self,
        table: nn.Parameter,
        columns: slice,
        auxiliary: Iterable[nn.Parameter] = (),
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        chunk_elements: int = 1048576,
    ) -> None:
        if table.ndim != 2 or columns.step not in (None, 1):
            raise ValueError("partition optimizer requires a 2D table and contiguous columns")
        start = 0 if columns.start is None else int(columns.start)
        stop = table.shape[1] if columns.stop is None else int(columns.stop)
        if (
            not 0 <= start < stop <= table.shape[1]
            or lr <= 0
            or eps <= 0
            or chunk_elements <= 0
        ):
            raise ValueError("partition optimizer configuration differs")
        self.table = table
        self.columns = slice(start, stop)
        self.lr = float(lr)
        self.beta1, self.beta2 = (float(value) for value in betas)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.chunk_elements = int(chunk_elements)
        self.step_count = 0
        owned = table.detach()[:, self.columns]
        self.first = torch.zeros_like(owned)
        self.second = torch.zeros_like(owned)
        parameters = [value for value in auxiliary if value.requires_grad]
        self.auxiliary = (
            torch.optim.AdamW(
                parameters,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
            if parameters
            else None
        )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.table.grad = None if set_to_none else torch.zeros_like(self.table)
        if self.auxiliary is not None:
            self.auxiliary.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        gradient = self.table.grad
        if gradient is None:
            raise RuntimeError("owned table has no gradient")
        value = self.table[:, self.columns]
        grad = gradient[:, self.columns]
        self.step_count += 1
        self.first.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
        self.second.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
        if self.weight_decay:
            value.mul_(1.0 - self.lr * self.weight_decay)
        correction1 = 1.0 - self.beta1**self.step_count
        correction2 = 1.0 - self.beta2**self.step_count
        rows_per_chunk = max(1, self.chunk_elements // value.shape[1])
        for start in range(0, value.shape[0], rows_per_chunk):
            stop = min(start + rows_per_chunk, value.shape[0])
            denominator = (
                self.second[start:stop].sqrt().div_(correction2**0.5).add_(self.eps)
            )
            value[start:stop].addcdiv_(
                self.first[start:stop], denominator, value=-self.lr / correction1
            )
        if self.auxiliary is not None:
            self.auxiliary.step()


__all__ = ["PartitionOwnedAdamW"]
