"""Authority, sparse transport and view grouping shared by typed providers.

No benchmark names, feature alignment, GT, implicit negative construction or
new confidence-product rule belongs here. Providers supply identity decisions
and signed domains; readouts retain their actual objectives.
"""
from __future__ import annotations

import atexit
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path

_COUNTS = Counter()
_NEGATIVE_ORIGINS = {'user_annotation', 'model_complement', 'category_competition',
                     'visible_region_complement', 'provider_declared'}


def enabled():
    mode = os.environ.get('RADIO_GS_EVIDENCE_CORE', 'legacy')
    if mode not in ('legacy', 'shared_v1'):
        raise ValueError('unknown RADIO_GS_EVIDENCE_CORE profile: ' + mode)
    return mode == 'shared_v1'


def _record(operation):
    _COUNTS[operation] += 1


def trace_summary():
    return dict(schema='radiogs.evidence_core_trace.v1',
                core='shared_v1', calls=dict(sorted(_COUNTS.items())),
                teacher_scores_calibrated=False, benchmark_specific_branches=False)


def _save_trace():
    directory = os.environ.get('RADIO_GS_EVIDENCE_TRACE_DIR')
    if directory and _COUNTS:
        root = Path(directory); root.mkdir(parents=True, exist_ok=True)
        path = root / ('core_%s.json' % os.getpid())
        path.write_text(json.dumps(trace_summary(), indent=2) + '\n')


atexit.register(_save_trace)


@dataclass(frozen=True)
class EvidenceDomains:
    positive: object
    negative: object
    unknown: object
    visible: object
    negative_origin: str


def _boolean(value):
    # Support NumPy and Torch masks without importing a GPU runtime for NumPy.
    return str(getattr(value, 'dtype', None)) in ('bool', 'torch.bool')


def _within_shape(value, shape):
    """Broadcasting may fill query columns, never create new observation axes."""
    dims = tuple(getattr(value, 'shape', ()))
    if len(dims) > len(shape) or any(a not in (1, b) for a, b in zip(dims[::-1], shape[::-1])):
        raise ValueError('evidence shape would expand the observation/query domain')


def compile_domains(positive, negative, *, visible, applicable=None,
                    unknown=None, exclusion_veto=None, negative_origin):
    """Apply explicit authority; unknown is never converted to a negative.

    'unknown' removes BOTH signs. 'exclusion_veto' removes only exclusions:
    a rejected proposal may overlap an independently admitted positive region.
    Those two cases must not be conflated.
    """
    if negative_origin not in _NEGATIVE_ORIGINS:
        raise ValueError('negative evidence requires a declared origin')
    if tuple(positive.shape) != tuple(negative.shape):
        raise ValueError('positive and negative evidence shapes differ')
    for mask in (positive, negative, visible, applicable, unknown, exclusion_veto):
        if mask is not None and not _boolean(mask):
            raise ValueError('authority and observation domains must be boolean')
        if mask is not None:
            _within_shape(mask, positive.shape)
    p = positive & visible
    n = negative & visible
    if applicable is not None:
        p = p & applicable; n = n & applicable
    if unknown is not None:
        p = p & ~unknown; n = n & ~unknown
    if exclusion_veto is not None:
        n = n & ~exclusion_veto
    if bool((p & n).any()):
        raise ValueError('one observation cannot assert both signs')
    u = visible & ~(p | n)
    _record('compile_domains')
    _record('negative_origin:' + negative_origin)
    return EvidenceDomains(p, n, u, visible, negative_origin)


def signed_weights(domains, positive_strength, negative_strength):
    """Provider strengths, not products of alleged correctness probabilities."""
    for value in (positive_strength, negative_strength):
        _within_shape(value, domains.positive.shape)
    _record('signed_weights')
    return (domains.positive * positive_strength,
            domains.negative * negative_strength)


def project(operator, values):
    """Identical sparse transport for contributions, masses and response values."""
    if operator.ndim != 2 or operator.shape[1] != values.shape[0]:
        raise ValueError('evidence operator and observation domain differ')
    _record('project')
    if hasattr(operator, 'layout') and values.ndim == 2:
        import torch
        if operator.layout in (torch.sparse_coo, torch.sparse_csr):
            return torch.sparse.mm(operator, values)
    return operator @ values


def view_balanced_transfer(membership, offsets):
    """Registered-map normalization: masks share a view, supported views share a carrier.

    This is not Gaussian opacity/contribution-mass normalization. The distinction
    is part of the adapter contract and is not inferred from a benchmark name.
    """
    import numpy as np
    from scipy import sparse
    offsets = np.asarray(offsets)
    if (offsets.ndim != 1 or len(offsets) < 2 or offsets.dtype.kind not in 'iu' or
            offsets[0] != 0 or offsets[-1] != membership.shape[1] or (np.diff(offsets) < 0).any()):
        raise ValueError('invalid region offsets for view-balanced transfer')
    blocks = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        block = membership[:, lo:hi].tocsr()
        mass = np.asarray(block.sum(1)).ravel()
        blocks.append(sparse.diags(1 / np.maximum(mass, 1)) @ block)
    transfer = sparse.hstack(blocks, format='csr')
    mass = np.asarray(transfer.sum(1)).ravel()
    _record('view_balanced_transfer')
    return (sparse.diags(1 / np.maximum(mass, 1)) @ transfer).tocsr()


def grouped_membership_features(membership, visibility, region_frames, anchor, vertices):
    """Keep within-view nesting/disagreement before the complete member head.

    The operation order and final float32 cast match the retained implementation.
    All-mask/any-mask fractions are not replaced with a single mean response.
    """
    import numpy as np
    from scipy import sparse
    regions = membership[anchor].indices
    frames, inverse, counts = np.unique(region_frames[regions], return_inverse=True, return_counts=True)
    association = sparse.csc_matrix((1. / counts[inverse],
                                    (np.arange(len(regions)), inverse)),
                                   shape=(len(regions), len(frames)))
    votes = project(membership[vertices][:, regions], association).tocsr()
    known = np.asarray(visibility[vertices][:, frames].sum(1)).ravel()
    mass = np.asarray(votes.sum(1)).ravel()
    any_mask = np.asarray((votes > 0).sum(1)).ravel()
    all_masks = np.asarray((votes >= 1. - 1e-6).sum(1)).ravel()
    denominator = np.maximum(known, 1)
    _record('grouped_membership_features')
    return np.column_stack([
        np.clip(mass / denominator, 0, 1), any_mask / denominator,
        all_masks / denominator, (any_mask - all_masks) / denominator,
        known / max(len(frames), 1), known > 0,
    ]).astype(np.float32)


def scope_response(membership, visibility, weights, confidence, *,
                   region_frames, completeness=None):
    """View-exclusive region support and declared visible-complement exclusion.

    This preserves the existing text scope policy. Predicted complements remain
    fallible; an unsupported element has no licensed exclusion observation.
    """
    import numpy as np
    positive = np.asarray(project(membership, weights)).reshape(-1)
    known = np.asarray(project(visibility, confidence)).reshape(-1)
    if completeness is not None:
        exclusion = np.asarray(project(visibility, confidence * completeness)).reshape(-1)
        exclusion -= np.asarray(project(membership, weights * completeness[region_frames])).reshape(-1)
        known = positive + np.maximum(exclusion, 0)
    probability = np.divide(positive, known, out=np.zeros_like(positive), where=known > 0)
    _record('scope_response')
    _record('negative_origin:visible_region_complement')
    return probability, known
