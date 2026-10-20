"""Transient membership hypotheses over the SAME compact multi-teacher field.

The scalar native maximum proposes a hypothesis; it does not certify an object.
SAM-guided membership suppresses duplicate proposals and pools native evidence.
No learned state, source cache, proposal bank, or teacher field is added.
"""
from __future__ import annotations
import torch

from .contracts import QueryResult, Support


class HypothesisRuntime:
    def __init__(self, field, rule, *, count=8, radius_m=.05, margin=.02):
        if rule not in {"peak", "pooled"}:
            raise ValueError("unknown hypothesis ranking rule")
        if count < 1 or radius_m <= 0 or margin < 0:
            raise ValueError("invalid hypothesis policy")
        self.field=field
        self.rule=rule;self.count=count;self.radius_m=radius_m;self.margin=margin

    def __getattr__(self, key):
        return getattr(self.field,key)

    @torch.no_grad()
    def query(self, packet):
        field=self.field
        packet.validate(len(field.xyz),field.semantic.native_dim,field.local.native_dim)
        if packet.positive is not None:
            # Legal ray/3D/signed supports use the exact existing member core.
            return field.query(packet)
        scale=field.default_scale if packet.scale is None else packet.scale
        native=field.semantic.score(packet.semantic).flatten()
        identity=native
        if packet.local is not None:
            patches=field.local.score(packet.local)
            weights=patches.new_ones(patches.shape[1]) if packet.local_weights is None else packet.local_weights.to(patches)
            identity=.5*(native+(patches*(weights/weights.sum())[None]).sum(1))
        eligible=field.observed.clone()
        if not eligible.any():raise ValueError("no observed identity candidates")
        ids=[];alternatives=[];pooled=[];sizes=[]
        for _ in range(self.count):
            if not eligible.any():break
            index=identity.masked_fill(~eligible,-torch.inf).argmax()
            support=Support(index.reshape(1),torch.ones(1,device=index.device))
            member=field.support_membership(support,scale,packet.negative)
            selected=(member>field.threshold)&field.observed
            # Empty hypotheses remain empty. A missing predicted support cannot
            # manufacture evidence; fall back to its proposal score for ranking.
            pool=(identity[selected]*member[selected]).sum()/member[selected].sum() if selected.any() else identity[index]
            ids.append(index);alternatives.append(member);pooled.append(pool);sizes.append(int(selected.sum()))
            # Distinct hypotheses, not spatially averaged seeds. Physical NMS
            # provides a stable fallback even if the member prediction is empty.
            near=(field.xyz-field.xyz[index]).norm(dim=1)<self.radius_m*field.world_units_per_metre
            eligible &= ~(selected|near)
        candidates=torch.stack(ids)
        pooled=torch.stack(pooled)
        peak=identity[candidates]
        ranking=peak if self.rule=="peak" else pooled
        if packet.task=="instance":
            admitted=torch.zeros(len(candidates),dtype=torch.bool,device=candidates.device)
            admitted[ranking.argmax()]=True
        else:
            admitted=ranking>=ranking.max()-self.margin
        members=torch.stack([p for i,p in enumerate(alternatives) if admitted[i]],dim=1).max(1).values
        weights=(10*ranking).softmax(0)
        return QueryResult(native,members,members>field.threshold,candidates,weights,scale,
            dict(architecture="unified_compact_v1",query_mechanism="transient_member_hypotheses_v1",
                 rule=self.rule,single_latent=True,source_access=False,persistent_proposal_bank=False,
                 candidate_count=len(candidates),candidate_radius_m=self.radius_m,identity_margin=self.margin,
                 peak_scores=peak.cpu().tolist(),pooled_scores=pooled.cpu().tolist(),
                 hypothesis_sizes=sizes,admitted=admitted.cpu().tolist(),hard_positive_constraint=False))
