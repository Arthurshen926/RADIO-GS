"""Response-evidence permissions, independent of dataset names and teachers."""
from __future__ import annotations

from dataclasses import replace
import torch
from .response_refinement import PartialResponseEvidence


def admit_response_observation(operator, target, positive, available_negative,
        confidence, *, exclusion=None, compatible=None, normalization_domain=None):
    """Return partial-evidence and complement-as-negative matched observations.

    available_negative marks visible, non-boundary, non-rejected locations.
    Merely being in this domain does not grant exclusion. The provider must
    supply a separate exclusion witness (reference mask or exclusive category).
    The control changes only that permission; targets and weights are identical.
    """
    if positive.dtype != torch.bool or available_negative.dtype != torch.bool:
        raise ValueError('positive/available-negative domains must be boolean')
    if positive.shape != target.shape or available_negative.shape != target.shape:
        raise ValueError('domains must have the target shape')
    if bool((positive & available_negative).any()):
        raise ValueError('positive and available-negative domains overlap')
    if exclusion is None:
        exclusion = torch.zeros_like(available_negative)
    if exclusion.dtype != torch.bool or exclusion.shape != target.shape:
        raise ValueError('exclusion witness must have boolean target shape')
    if compatible is None:
        compatible = torch.ones((), dtype=torch.bool, device=target.device)
    if normalization_domain is None:
        normalization_domain = positive | available_negative
    item = PartialResponseEvidence(operator, target, positive,
        available_negative & exclusion, compatible, confidence, normalization_domain)
    return item, replace(item, negative=available_negative)


def exclusive_category_exclusion(positive):
    """Only an unambiguous rival foreground licenses category exclusion.

    A provider must call this on a complete mutually exclusive menu. Overlapping
    labels, parts and multiple simultaneous positive categories do not qualify.
    """
    if positive.ndim != 2 or positive.dtype != torch.bool:
        raise ValueError('expected pixel-by-category boolean foreground')
    unique = positive.sum(1, keepdim=True) == 1
    return unique & ~positive
