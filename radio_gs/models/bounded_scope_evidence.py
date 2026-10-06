"""Finite structural evidence cannot override arbitrarily strong native logits."""
import math
import torch


def bounded_scope_logits(native_logits, membership, strength, *, suppress_outside=True):
    if native_logits.shape != membership.shape or membership.dtype != torch.bool:
        raise ValueError('Aligned native logits and boolean scope membership required')
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('Scope strength must be finite and nonnegative')
    evidence = 2 * membership.to(native_logits) - 1 if suppress_outside else membership.to(native_logits)
    return native_logits + strength * evidence
