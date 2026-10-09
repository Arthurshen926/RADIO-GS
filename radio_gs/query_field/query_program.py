"""One source-free query program with typed native evidence and common members."""
import torch

from .category_evidence import corroborated_category_scores
from .contracts import QueryPacket, QueryResult, Support
from .identity_evidence import native_rank_fusion, bidirectional_local_match

PROGRAM = 'native_evidence_shared_members_v2'


@torch.no_grad()
def query(field, packet):
    """Native adapters select identity; every spatial output uses the same field.

    Pure reference-image instances retain global ordinal candidates and local
    set correspondence. Registered supports, text unions and explicit image
    unions retain their existing typed conditions. No per-query fallback to an
    old solver, source observation bank or benchmark-selected winner exists.
    """
    packet.validate(len(field.xyz), field.semantic.native_dim, field.local.native_dim)
    if (packet.modality != 'image' or packet.local is None or packet.positive is not None
            or packet.task != 'instance'):
        return field.query(packet)
    scale = field.default_scale if packet.scale is None else packet.scale
    native = field.semantic.score(packet.semantic).flatten()
    local = field.local.score(packet.local, patch_max=True)
    available = field.observed if field.anchor_eligible is None else field.anchor_eligible
    global_identity = native_rank_fusion(native, local, available)
    candidates = field._candidates(global_identity)
    correspondence = torch.zeros(len(candidates), device=field.xyz.device)
    valid = torch.zeros(len(candidates), dtype=torch.bool, device=field.xyz.device)
    logits, details = [], []
    for i, candidate in enumerate(candidates):
        support = Support(candidate.reshape(1), torch.ones(1, device=field.xyz.device))
        energy = field.member_logits(support, scale)
        logits.append(energy)
        rows = torch.where((energy.sigmoid() > field.threshold) & field.observed)[0]
        if not len(rows):
            details.append(dict(anchor=int(candidate), elements=0))
            continue
        match = bidirectional_local_match(field.local, rows, packet.local.reshape(-1, field.local.native_dim))
        correspondence[i] = match.score; valid[i] = True
        details.append(dict(anchor=int(candidate), elements=match.field_elements,
                            forward=float(match.forward), reverse=float(match.reverse)))
    if bool(valid.any()):
        identity = native_rank_fusion(native[candidates], correspondence, valid).masked_fill(~valid, -torch.inf)
        weights = torch.softmax(10 * identity, dim=0)
        chosen = int(weights.argmax())
        membership = field.exclude_membership(logits[chosen].sigmoid(), packet.negative, scale, logits[chosen])
        membership = membership.masked_fill(~field.member_available, 0)
    else:
        candidates = candidates[:0]
        weights = torch.empty(0, device=field.xyz.device)
        membership = torch.zeros(len(field.xyz), device=field.xyz.device)
    return QueryResult(native, membership, membership > field.threshold, candidates, weights, float(scale),
        dict(query_program=PROGRAM, shared_member_core=True, source_access=False,
             conditional_relation_decoder=field.relation_decoder,
             identity_rule='global native ranks, field-set bidirectional correspondence, candidate native ranks',
             uncertainty_resolved=False, membership_scores_calibrated=False,
             hard_positive_constraint=False, local_correspondence=details,
             source_patch_bank_reproduced=False, empty_scope_policy='empty output, no forced anchor'))


@torch.no_grad()
def category_partition(field, native_queries):
    """Complete category menu; native unaries and members come from one field."""
    native, members = [], []
    for vector in native_queries:
        result = query(field, QueryPacket('text', task='union', semantic=vector))
        native.append(result.semantic_scores); members.append(result.membership_scores)
    native = torch.stack(native, 1); members = torch.stack(members, 1)
    evidence = corroborated_category_scores(native, members)
    labels = evidence.argmax(1).masked_fill(~field.observed, -1)
    return evidence, labels
