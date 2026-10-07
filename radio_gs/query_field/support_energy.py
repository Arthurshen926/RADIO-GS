"""Sufficient moments of uncertain supports for the common member energy.

This is an expected *logit*, not an expectation of calibrated probabilities.
Positive co-target observations use their weighted mean energy. Independent
negative rays use the maximum exclusion response: a duplicate constraint has
no extra effect, and unrelated backgrounds cannot cancel into one prototype.
"""
import torch
from torch.nn import functional as F


def moments(field, features, indices, weights):
    weights = weights / weights.sum(-1, keepdim=True)
    group = (features[indices] * weights[..., None]).sum(-2)
    positions = field.xyz[indices]
    centre = (positions * weights[..., None]).sum(-2)
    variance = ((positions-centre[..., None, :]).square().sum(-1)*weights).sum(-1)
    local = None
    if field.local_metric == 'native_cosine':
        local = (F.normalize(field.local.decode(indices), dim=-1)*weights[..., None]).sum(-2)
    return group, centre, variance, local


def logits_from_moments(field, features, scale, statistics, rows=None):
    group, centre, variance, local = statistics
    selected = features if rows is None else features[rows]
    affinity = selected @ group.T - field.affinity_bias
    if local is not None:
        # NativeCoordinates.score normalizes its query: restore the resultant
        # norm so this remains E[cosine], including a zero resultant.
        norm = local.norm(dim=-1)
        if rows is None:
            cosine = field.local.score(local) * norm
        else:
            cosine = F.normalize(field.local.decode(rows), dim=-1) @ local.T
        affinity = affinity - (1-cosine.clamp(-1, 1))
    logits = field.log_temperature.exp().clamp(1, 50) * affinity
    if field.spatial_metric == 'scale_normalized':
        xyz = field.xyz if rows is None else field.xyz[rows]
        distance_squared = (xyz[:, None, :] - centre[None, :, :]).square().sum(-1)
        logits = logits - (distance_squared + variance)/(2*(float(scale)*field.world_units_per_metre)**2)
    return logits


def expected_logits(field, support, scale, rows=None):
    features = field.grouping_features(scale)
    ids = support.indices.to(features.device)[None]
    weights = support.weights.to(features)[None]
    return logits_from_moments(field, features, scale, moments(field, features, ids, weights), rows)[:, 0]


def independent_exclusion(field, support, scale, chunk_size=32):
    """Exact max over all supported rays, chunked; no ray subsampling/top-k.

    Pad contiguous CPU ray gathers and reduce along rows on device. No CUDA
    scatter reduction is used to construct query moments. Padding has zero mass.
    Chunking controls transient memory only and is fixed for inference/replay.
    """
    components = support.components.detach().cpu()
    order = torch.argsort(components, stable=True)
    _, counts = torch.unique_consecutive(components[order], return_counts=True)
    ids = support.indices.detach().cpu()[order]
    weights = support.weights.detach().float().cpu()[order]
    starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    features = field.grouping_features(scale)
    exclusion = torch.zeros(len(field.xyz), device=features.device)
    for start in range(0, len(counts), chunk_size):
        sizes = counts[start:start+chunk_size]
        padded_ids = torch.zeros(len(sizes), int(sizes.max()), dtype=torch.long)
        padded_weights = torch.zeros_like(padded_ids, dtype=torch.float32)
        for offset, size in enumerate(sizes.tolist()):
            a, b = starts[start+offset:start+offset+2].tolist()
            padded_ids[offset, :size] = ids[a:b]
            padded_weights[offset, :size] = weights[a:b]
        statistics = moments(field, features, padded_ids.to(features.device), padded_weights.to(features.device))
        value = logits_from_moments(field, features, scale, statistics).max(dim=1).values.sigmoid()
        exclusion = torch.maximum(exclusion, value)
    return exclusion
