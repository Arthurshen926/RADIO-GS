"""Query-only identity/scope hypotheses over one unchanged member operator.

No observation caches or target masks are accepted by this module. The native
bidirectional score is a fixed research control, not calibrated correctness.
"""
from dataclasses import dataclass, replace
import math

import torch
from torch.nn import functional as F

from .contracts import QueryPacket, QueryResult, Support


@dataclass
class MemberHypothesis:
    anchor: int
    scale: float
    membership: torch.Tensor
    selected: torch.Tensor
    anchor_score: float
    native_score: float
    evidence: dict


def reverse_local_match(local, query, rows, chunk_size=8192):
    """Every query patch finds its best native match, with bounded scene memory."""
    query = F.normalize(query.to(local.codes.device).float().reshape(-1,local.native_dim), dim=-1)
    best = torch.full((len(query),), -torch.inf, device=query.device)
    for block in rows.split(chunk_size):
        values = F.normalize(local.decode(block),dim=-1)
        best = torch.maximum(best, (values@query.T).max(0).values)
    return float(best.mean()) if len(rows) else -1.


@torch.no_grad()
def propose(field, packet: QueryPacket, scale_factors=(1., .5, 2.), *, score_native=True):
    """Enumerate before selection; explicitly supplied scales are never replaced."""
    packet.validate(len(field.xyz),field.semantic.native_dim,field.local.native_dim)
    if packet.task != "instance":
        raise ValueError("instance hypotheses do not define category unions")
    if not scale_factors or any(not math.isfinite(s) or s<=0 for s in scale_factors):
        raise ValueError("scale factors must be finite and positive")
    scales = [float(packet.scale)] if packet.scale is not None else [field.default_scale*s for s in scale_factors]
    if packet.positive is not None:
        hypotheses=[]
        for scale in scales:
            result=field.query(replace(packet,scale=scale))
            hypotheses.append(MemberHypothesis(-1,scale,result.membership_scores,result.selected,0.,0.,
                                               dict(registered_support=True,**result.diagnostics)))
        return hypotheses
    semantic=field.semantic.score(packet.semantic).flatten()
    local=None
    if packet.local is not None:
        local=field.local.score(packet.local,patch_max=True)
    identity=semantic if local is None else (semantic+local)*.5
    anchors=field._candidates(identity)
    hypotheses=[]
    # Scale-major order leaves the old default hypothesis at index zero.
    for scale in scales:
        for anchor in anchors:
            support=Support(anchor.reshape(1),torch.ones(1,device=field.xyz.device))
            membership=field.member_logits(support,scale).sigmoid()
            if packet.negative is not None:
                ids=packet.negative.indices.to(field.xyz.device)
                weights=packet.negative.weights.to(field.xyz.device)
                good=field.observed[ids]&(weights>0)
                if bool(good.any()):
                    membership*=1-field.member_logits(Support(ids[good],weights[good]),scale).sigmoid()
            membership=membership.masked_fill(~field.observed,0)
            selected=membership>field.threshold
            rows=torch.where(selected)[0]
            evidence=dict(selected_elements=len(rows),registered_support=False)
            native_score=-1e6
            if len(rows) and score_native:
                semantic_mean=float(semantic[rows].mean())
                evidence['semantic_mean']=semantic_mean
                native_score=semantic_mean
                if local is not None:
                    forward=float(local[rows].mean())
                    reverse=reverse_local_match(field.local,packet.local,rows)
                    evidence.update(local_forward=forward,local_reverse=reverse)
                    native_score=.5*semantic_mean+.25*(forward+reverse)
            hypotheses.append(MemberHypothesis(int(anchor),float(scale),membership,selected,
                                                float(identity[anchor]),native_score,evidence))
    return hypotheses


def choose(hypotheses, policy="anchor"):
    if not hypotheses:
        raise ValueError("no supported member hypotheses")
    if policy=="anchor":
        return 0
    if policy=="native_bidirectional":
        if hypotheses[0].evidence['registered_support']:
            raise ValueError("native identity selection requires native query evidence")
        return max(range(len(hypotheses)),key=lambda i:hypotheses[i].native_score)
    raise ValueError("unknown hypothesis selection policy")


@torch.no_grad()
def query(field, packet, *, policy="anchor", scale_factors=(1., .5, 2.)):
    if packet.positive is not None:
        if policy!='anchor':
            raise ValueError("native identity selection requires native query evidence")
        return field.query(packet)
    hypotheses=propose(field,packet,scale_factors)
    chosen=choose(hypotheses,policy)
    hypothesis=hypotheses[chosen]
    scores=torch.tensor([h.anchor_score if policy=='anchor' else h.native_score for h in hypotheses],
                         device=field.xyz.device)
    return QueryResult(None if packet.semantic is None else field.semantic.score(packet.semantic).flatten(),
        hypothesis.membership,hypothesis.selected,
        torch.tensor([h.anchor for h in hypotheses],device=field.xyz.device),torch.softmax(scores*10,0),
        hypothesis.scale,dict(shared_member_core=True,source_access=False,selection_policy=policy,
            hypothesis_scales=[h.scale for h in hypotheses],selected_hypothesis=chosen,
            hypotheses=len(hypotheses),uncertainty_resolved=False,membership_scores_calibrated=False,
            identity_scores_calibrated=False,hard_positive_constraint=packet.exact_positive))
