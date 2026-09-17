"""Query-level admission from native/extent agreement, never query identity."""
import torch
from torch import nn


def extent_agreement_features(cosine, threshold, extent_logits, observed, concept_accepted):
    if cosine.shape != extent_logits.shape or observed.shape != cosine.shape[:1]:
        raise ValueError('Full-field native and extent axes required')
    margin = cosine - threshold
    positive = (margin > 0) & observed[:, None]
    extent = (extent_logits > 0) & observed[:, None]
    intersection = (positive & extent).sum(0).float()
    nm, em = positive.sum(0).float(), extent.sum(0).float()
    outside = observed[:, None] & ~extent
    peak = margin.masked_fill(~observed[:, None], -torch.inf).amax(0)
    peak = torch.where(observed.any(), peak, torch.zeros_like(peak))
    return torch.stack([
        intersection / (nm + em - intersection).clamp_min(1),
        intersection / em.clamp_min(1), intersection / nm.clamp_min(1),
        torch.log((em + 1) / (nm + 1)),
        (margin * extent).sum(0) / em.clamp_min(1),
        (margin * outside).sum(0) / outside.sum(0).clamp_min(1),
        peak, concept_accepted.float(),
    ], 1)


class ExtentBenefitGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('center', torch.zeros(8))
        self.register_buffer('scale', torch.ones(8))
        self.register_buffer('coefficient', torch.zeros(9))

    def forward(self, cosine, threshold, extent_logits, observed, concept_accepted, *, query=None, head=None):
        x = extent_agreement_features(cosine, threshold, extent_logits, observed, concept_accepted)
        z = (x - self.center) / self.scale
        # Exact ties retain native output.
        return z @ self.coefficient[:-1] + self.coefficient[-1] > 0


class ExtentUtilityTree(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('feature',torch.full((7,),-1,dtype=torch.long))
        self.register_buffer('cutoff',torch.zeros(7))
        self.register_buffer('decision',torch.zeros(7,dtype=torch.bool))

    def decide_features(self, x):
        node=torch.zeros(len(x),device=x.device,dtype=torch.long)
        for _ in range(2):
            feature=self.feature[node];active=feature>=0
            right=x.gather(1,feature.clamp_min(0)[:,None])[:,0]>self.cutoff[node]
            node=torch.where(active,2*node+1+right.long(),node)
        return self.decision[node]

    def forward(self, cosine, threshold, extent_logits, observed, concept_accepted, *, query=None, head=None):
        return self.decide_features(extent_agreement_features(
            cosine,threshold,extent_logits,observed,concept_accepted))


def scope_semantic_features(query, text_prototype, image_prototype, concept_accepted):
    from torch.nn import functional as F
    query=F.normalize(query,dim=-1)
    values=torch.stack([(query*F.normalize(text_prototype,dim=-1)).sum(-1),
                        (query*image_prototype).sum(-1)],1)
    # Rejected concepts do not describe the selected fallback region.
    return torch.where(concept_accepted[:,None],values,torch.zeros_like(values))


class SemanticExtentBenefitGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('center',torch.zeros(10))
        self.register_buffer('scale',torch.ones(10))
        self.register_buffer('coefficient',torch.zeros(11))

    def forward(self, cosine, threshold, extent_logits, observed, concept_accepted, *, query=None, head=None):
        if query is None or head is None or not hasattr(head,'routing_scope_images'):
            raise ValueError('Semantic scope admission requires source text/image prototypes')
        index,_,_=head.route(query)
        agreement=extent_agreement_features(cosine,threshold,extent_logits,observed,concept_accepted)
        semantic=scope_semantic_features(query,head.prototypes[index],head.routing_scope_images[index],concept_accepted)
        z=(torch.cat([agreement,semantic],1)-self.center)/self.scale
        return z@self.coefficient[:-1]+self.coefficient[-1]>0
