"""Relative query supervision from disjoint, teacher-separated source scopes.

Unknown background, containment, overlapping parts and aliases supply no
negative relation. This is source teacher evidence, not physical identity GT.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def disjoint_query_pairs(masks, descriptors, queries):
    if masks.ndim != 2 or descriptors.ndim != 2 or queries.shape != descriptors.shape:
        raise ValueError('Aligned scope masks and native descriptor/query vectors required')
    if len(masks) != len(descriptors) or not torch.isfinite(descriptors).all() or not torch.isfinite(queries).all():
        raise ValueError('Finite aligned source evidence required')
    if (descriptors.norm(dim=-1) == 0).any() or (queries.norm(dim=-1) == 0).any():
        raise ValueError('Nonzero native source evidence required')
    membership = masks.bool().float()
    overlap = membership @ membership.T
    similarity = F.normalize(descriptors.float(), dim=-1) @ F.normalize(queries.float(), dim=-1).T
    preference = similarity.diag()[:, None] > similarity
    nonempty = membership.sum(1) > 0
    eligible = (overlap == 0) & preference & preference.T & nonempty[:, None] & nonempty[None]
    return eligible.triu(diagonal=1).nonzero()


def disjoint_query_order_loss(probability, masks, valid):
    """Zero-margin hinge on two queries, only within their positive evidence.

    probability has pixel/query axes. For each pixel in scope i, q_i should
    exceed q_j. No targets are constructed outside both source scopes.
    """
    if probability.ndim != 2 or probability.shape[1] != 2 or masks.shape != probability.T.shape:
        raise ValueError('Exactly two source queries with pixel-aligned masks required')
    if valid.shape != (len(probability),) or not torch.isfinite(probability).all():
        raise ValueError('Finite pixel responses and aligned support required')
    if (masks[0].bool() & masks[1].bool()).any():
        raise ValueError('Overlapping scopes cannot define exclusive query competition')
    if not all((masks[i].bool() & valid.bool()).any() for i in (0, 1)):
        return probability.sum() * 0
    pieces = []
    for i in (0, 1):
        keep = masks[i].bool() & valid.bool()
        if keep.any():
            pieces.append((probability[keep, 1-i] - probability[keep, i]).clamp_min(0).mean())
    return torch.stack(pieces).mean() if pieces else probability.sum() * 0


@torch.no_grad()
def disjoint_query_order_accuracy(probability, masks, valid):
    if not all((masks[i].bool() & valid.bool()).any() for i in (0, 1)):
        return None
    values = []
    for i in (0, 1):
        keep = masks[i].bool() & valid.bool()
        if keep.any():
            values.append((probability[keep, i] > probability[keep, 1-i]).float().mean())
    return float(torch.stack(values).mean()) if values else None
