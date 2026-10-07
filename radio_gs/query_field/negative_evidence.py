"""Relative signed evidence through the same conditional member logits.

Positive, explicit negative and null scores compete as exp(l+), exp(l-) and 1.
Ray mixtures average negative evidence by its observed mass before competition;
they are neither independent vetoes nor calibrated class probabilities.
"""
import torch
from torch.nn import functional as F

from .support_energy import moments, logits_from_moments


def competitive_membership(positive_logits, negative_logits):
    # exp(l+) / (1 + exp(l+) + exp(l-)), without exponent overflow or
    # reconstructing saturated positive logits from their rounded sigmoid.
    return torch.sigmoid(positive_logits-F.softplus(negative_logits))


def ray_log_mixture(field, support, scale, chunk_size=32):
    """Mass-normalized mixture of unchanged per-ray prototype member energies.

    Every supported ray participates. The caller filters unobserved/zero mass.
    Splitting a ray into identical fractional observations or repeating the
    entire evidence measure leaves the algebra unchanged. This does not promise
    invariance when a different raster changes actual geometric associations.
    """
    if support.components is None:
        return field.member_logits(support, scale)
    components = support.components.detach().cpu()
    order = torch.argsort(components, stable=True)
    _, counts = torch.unique_consecutive(components[order], return_counts=True)
    indices = support.indices.detach().cpu()[order]
    weights = support.weights.detach().float().cpu()[order]
    starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    total = weights.double().sum()
    features = field.grouping_features(scale)
    accumulated = torch.full((len(features),), -torch.inf, device=features.device)
    for start in range(0, len(counts), chunk_size):
        sizes = counts[start:start+chunk_size]
        ids = torch.zeros(len(sizes), int(sizes.max()), dtype=torch.long)
        mass = torch.zeros_like(ids, dtype=torch.float32)
        for offset, size in enumerate(sizes.tolist()):
            a, b = starts[start+offset:start+offset+2].tolist()
            ids[offset, :size] = indices[a:b]
            mass[offset, :size] = weights[a:b]
        prior = (mass.double().sum(1)/total).log().to(features)
        group, centre, _, local = moments(field, features, ids.to(features.device), mass.to(features.device))
        # Restore the accepted per-ray prototype algebra: normalize directions
        # and use the centre prior, with no expected-energy variance penalty.
        statistics = (F.normalize(group, dim=-1), centre, torch.zeros(len(sizes), device=features.device),
                      None if local is None else F.normalize(local, dim=-1))
        logits = logits_from_moments(field, features, scale, statistics)
        accumulated = torch.logaddexp(accumulated, torch.logsumexp(logits+prior, dim=1))
    return accumulated
