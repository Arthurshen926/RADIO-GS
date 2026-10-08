"""Native identity evidence over a Gaussian field; no source-memory reader.

These primitives retain v10's ordinal fusion and bidirectional local matching.
A set of fused Gaussian descriptors is not equivalent to v10's original source
patch set. Whether it retains that capability is an experimental question.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .contracts import QueryPacket, Support

IDENTITY_CONTROLS = ('parent', 'rank_same_candidates', 'bidirectional_same_candidates',
                     'rank_global_candidates')


def percentile_rank(values, available):
    """One-based average ranks, including ties, over the declared support only."""
    if values.ndim != 1 or available.shape != values.shape or available.dtype != torch.bool:
        raise ValueError("ranks require a vector and an aligned Boolean support")
    indices = torch.where(available)[0]
    if not len(indices) or not bool(torch.isfinite(values[indices]).all()):
        raise ValueError("rank support must be nonempty and finite")
    ordered, permutation = torch.sort(values[indices], stable=True)
    _, inverse, counts = torch.unique_consecutive(ordered, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0)
    means = (ends.float() - (counts.float() - 1) * .5) / len(indices)
    result = torch.zeros_like(values, dtype=torch.float32)
    result[indices[permutation]] = means[inverse]
    return result


def native_rank_fusion(semantic, local, available):
    """Equal ordinal evidence; the result is not a foreground probability."""
    if semantic.shape != local.shape:
        raise ValueError("native identity channels must align")
    return .5 * (percentile_rank(semantic, available) + percentile_rank(local, available))


@dataclass(frozen=True)
class LocalCorrespondence:
    forward: torch.Tensor
    reverse: torch.Tensor
    query_patches: int
    field_elements: int

    @property
    def score(self):
        return .5 * (self.forward + self.reverse)


@torch.no_grad()
def bidirectional_local_match(coordinates, rows, query, *, field_weights=None,
                              query_weights=None, chunk_size=2048):
    """Weighted Chamfer cosine with every supported row and query patch.

    Chunking controls memory only. Affine native codecs are decoded with their
    bias and norm before matching. Zero-weight entries never supply a match.
    Duplicating a descriptor while splitting its mass leaves the score intact.
    """
    device = coordinates.codes.device
    rows = torch.as_tensor(rows, device=device)
    query = torch.as_tensor(query, device=device).float()
    if (rows.ndim != 1 or rows.dtype != torch.long or not len(rows)
            or bool(((rows < 0) | (rows >= len(coordinates.codes))).any())):
        raise ValueError("local matching requires valid nonempty field rows")
    if (query.ndim != 2 or query.shape[1] != coordinates.native_dim or not len(query)
            or not bool(torch.isfinite(query).all()) or bool((query.norm(dim=1) == 0).any())):
        raise ValueError("local query patches must be finite nonzero native vectors")
    if chunk_size < 1:
        raise ValueError("local matching chunk size must be positive")

    def mass(value, count):
        value = torch.ones(count, device=device) if value is None else torch.as_tensor(value, device=device).float()
        if (value.shape != (count,) or not bool(torch.isfinite(value).all())
                or bool((value < 0).any()) or not bool(value.sum() > 0)):
            raise ValueError("matching weights must have finite nonnegative positive mass")
        return value / value.sum()

    wm = mass(field_weights, len(rows))
    qm = mass(query_weights, len(query))
    rows, wm = rows[wm > 0], wm[wm > 0]
    query, qm = query[qm > 0], qm[qm > 0]
    query = F.normalize(query, dim=1)
    forward = torch.full((len(query),), -torch.inf, device=device)
    reverse = torch.zeros((), device=device)
    for start in range(0, len(rows), chunk_size):
        native = coordinates.decode(rows[start:start + chunk_size])
        if not bool(torch.isfinite(native).all()) or bool((native.norm(dim=1) == 0).any()):
            raise ValueError("supported field descriptors must be finite and nonzero")
        cosine = F.normalize(native, dim=1) @ query.T
        forward = torch.maximum(forward, cosine.max(0).values)
        reverse += (cosine.max(1).values * wm[start:start + chunk_size]).sum()
    return LocalCorrespondence((forward * qm).sum(), reverse, len(query), len(rows))


@torch.no_grad()
def image_identity_controls(field, semantic, local):
    """Four fixed research interventions; never changes the field or its default.

    All alternative outputs still come from the same frozen member operator.
    Same-candidate controls cannot repair missing identity/scope hypotheses.
    """
    device = field.xyz.device
    packet = QueryPacket('image', semantic=semantic.to(device), local=local.to(device))
    parent = field.query(packet)
    native = parent.semantic_scores
    local_peak = field.local.score(packet.local, patch_max=True)
    candidates = parent.candidate_indices
    values = {}

    def members(index):
        key = int(index)
        if key not in values:
            support = Support(index.reshape(1), torch.ones(1, device=device))
            values[key] = field.member_logits(support, parent.scale).sigmoid().masked_fill(~field.observed, 0)
        return values[key]

    def result(ids, evidence):
        chosen = ids[evidence.argmax()] if len(ids) else None
        membership = members(chosen) if chosen is not None else torch.zeros(len(field.xyz), device=device)
        return dict(membership=membership, candidates=ids, identity_scores=evidence,
                    chosen=-1 if chosen is None else int(chosen))

    outputs = {'parent': dict(membership=parent.membership_scores, candidates=candidates,
                             identity_scores=parent.candidate_weights,
                             chosen=int(candidates[parent.candidate_weights.argmax()]))}
    same_available = torch.ones(len(candidates), dtype=torch.bool, device=device)
    same_ranks = native_rank_fusion(native[candidates], local_peak[candidates], same_available)
    outputs['rank_same_candidates'] = result(candidates, same_ranks)
    correspondences = []
    local_set = torch.zeros(len(candidates), device=device)
    scope_available = torch.zeros(len(candidates), dtype=torch.bool, device=device)
    for i, anchor in enumerate(candidates):
        rows = torch.where(members(anchor) > field.threshold)[0]
        if not len(rows):
            correspondences.append(dict(anchor=int(anchor), elements=0, forward=None, reverse=None))
            continue
        match = bidirectional_local_match(field.local, rows, packet.local)
        local_set[i] = match.score
        scope_available[i] = True
        correspondences.append(dict(anchor=int(anchor), elements=match.field_elements,
                                    forward=float(match.forward), reverse=float(match.reverse)))
    if bool(scope_available.any()):
        ranks = native_rank_fusion(native[candidates], local_set, scope_available)
        outputs['bidirectional_same_candidates'] = result(candidates[scope_available], ranks[scope_available])
    else:
        outputs['bidirectional_same_candidates'] = result(candidates[:0], local_set[:0])
    available = field.observed if field.anchor_eligible is None else field.anchor_eligible
    global_ranks = native_rank_fusion(native, local_peak, available)
    global_candidates = field._candidates(global_ranks)
    outputs['rank_global_candidates'] = result(global_candidates, global_ranks[global_candidates])
    diagnostics = dict(scale_m=parent.scale, threshold=field.threshold,
                       source_access=False, shared_member_core=True,
                       calibrated_identity=False, field_tensors_changed=False,
                       correspondence=correspondences, query_patches=len(packet.local),
                       native_semantic_at_anchor=native[candidates].cpu().tolist(),
                       native_local_peak_at_anchor=local_peak[candidates].cpu().tolist(),
                       reference_patch_bank_reproduced=False)
    return outputs, diagnostics
