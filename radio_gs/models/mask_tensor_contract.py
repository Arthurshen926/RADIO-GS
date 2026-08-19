"""Fail-closed conversion for externally produced mask tensors.

Historical artifact keys such as ``mask_logits`` are not a sufficient tensor
contract: some official SAM3 APIs return probabilities under that key.  New
consumers must bind the numerical semantics explicitly before applying any
nonlinearity or threshold.
"""

from __future__ import annotations

import torch


MASK_TENSOR_PROBABILITY = "probability"
MASK_TENSOR_LOGIT = "logit"
SUPPORTED_MASK_TENSOR_SEMANTICS = frozenset(
    {MASK_TENSOR_PROBABILITY, MASK_TENSOR_LOGIT}
)


def mask_tensor_to_probability(
    value: torch.Tensor,
    *,
    semantics: str | None,
    label: str = "mask tensor",
) -> torch.Tensor:
    """Return mask probabilities only after validating explicit semantics.

    Missing or unknown semantics are rejected instead of guessed from a key or
    from the observed value range.  Range inference is ambiguous because a
    valid logit tensor may also lie entirely inside ``[0,1]``.
    """

    if not torch.is_tensor(value):
        raise ValueError(f"{label} must be a torch.Tensor")
    if semantics not in SUPPORTED_MASK_TENSOR_SEMANTICS:
        expected = ", ".join(sorted(SUPPORTED_MASK_TENSOR_SEMANTICS))
        raise ValueError(
            f"{label} requires explicit mask_tensor_semantics ({expected}); "
            f"got {semantics!r}"
        )
    tensor = value.detach().float()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} contains non-finite values")
    if semantics == MASK_TENSOR_PROBABILITY:
        if bool(((tensor < 0.0) | (tensor > 1.0)).any()):
            raise ValueError(f"{label} declared probability but is outside [0,1]")
        return tensor
    return torch.sigmoid(tensor)


def mask_tensor_to_binary(
    value: torch.Tensor,
    *,
    semantics: str | None,
    probability_threshold: float = 0.5,
    label: str = "mask tensor",
) -> torch.Tensor:
    """Threshold an explicitly typed mask tensor in probability space."""

    threshold = float(probability_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("probability_threshold must be in [0,1]")
    return mask_tensor_to_probability(
        value,
        semantics=semantics,
        label=label,
    ) >= threshold
