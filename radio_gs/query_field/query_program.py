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
    if field.native_modes is not None and packet.modality in {'text','image'} and packet.positive is None:
        from .native_modes import query_modes
        return query_modes(field,packet)
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
    if field.semantic_measure is None:
        evidence = corroborated_category_scores(native, members)
        available=field.observed
    else:
        from radio_gs.querying.observation_readout import positive_support_rank
        prior=field.semantic_measure.probabilities(native_queries,(100*native.float()).softmax(1),
                                                   field.semantic_measure_rule)
        evidence=positive_support_rank(prior,members)
        available=field.observed | field.semantic_measure.supported
    labels = evidence.argmax(1).masked_fill(~available, -1)
    return evidence, labels


@torch.no_grad()
def text_union(field, native_query, generic_negatives, *, rule='measure', temperature=10., corroborate=True, support_rule='positive_rank'):
    """Absolute native union on every supported mode, with common member support.

    This is an explicit opt-in typed interface. The generic negatives and
    temperature are query-program inputs, not stored historical class answers.
    """
    if support_rule not in {'positive_rank','admission_preserving'}:raise ValueError('unknown union support rule')
    if field.semantic_measure is None:raise ValueError('text union requires a semantic measure field')
    packet=QueryPacket('text',task='union',semantic=native_query)
    member=query(field,packet)
    background=field.semantic.score(generic_negatives).max(1).values
    fallback=(temperature*(member.semantic_scores-background)).sigmoid()
    prior=field.semantic_measure.union_probabilities(native_query,generic_negatives,fallback,rule,temperature)
    if corroborate:
        from radio_gs.querying.observation_readout import positive_support_rank
        if support_rule=='admission_preserving':
            probability=admission_preserving_support(prior,member.membership_scores,field.threshold)
        else:probability=positive_support_rank(prior,member.membership_scores)
    else:probability=prior
    available=field.observed|field.semantic_measure.supported
    probability=probability.masked_fill(~available,0)
    return QueryResult(prior,probability,probability>field.threshold,member.candidate_indices,
        member.candidate_weights,member.scale,dict(shared_member_core=True,source_access=False,
        typed_union=True,semantic_rule=rule,semantic_modes=len(field.semantic_measure.descriptors),
        semantic_candidate_cap=None,absolute_generic_admission=True,other_requests_compete=False,
        member_corroboration=corroborate,support_rule=support_rule,
        membership_extent_refined=False if support_rule=='admission_preserving' else None,
        member_candidate_count=len(member.candidate_indices),
        scores_calibrated=False,source_text_detection_transferred=False))


def admission_preserving_support(prior, member, threshold=.5):
    """Rank corroboration cannot turn rejected semantic evidence into admission.

    Positive semantic margin scales the existing bounded boost continuously.
    The selected set is exactly the semantic prior's set, not a claim of newly
    learned conditional extent. No benchmark-specific parameter is introduced.
    """
    if not 0 < threshold < 1:raise ValueError('union threshold must lie inside (0,1)')
    margin=((prior-threshold)/(1-threshold)).clamp(0,1)
    return prior+prior*(1-prior)*member*margin
