"""Preserve every positive ray hypothesis before nonlinear member aggregation."""
import torch
from torch.nn import functional as F
from .contracts import Support,QueryResult
from .support_energy import moments,logits_from_moments


def mixture_logits(field,support,scale,*,shared_centre=False,chunk_size=32,aggregation="mixture"):
    if aggregation not in {"mixture","union","confidence_union"}:raise ValueError("unknown positive aggregation")
    if support.components is None:return field.member_logits(support,scale,include_evidence=False)
    order=torch.argsort(support.components.cpu(),stable=True)
    _,counts=torch.unique_consecutive(support.components.cpu()[order],return_counts=True)
    ids=support.indices.cpu()[order];weights=support.weights.float().cpu()[order]
    starts=torch.cat([torch.zeros(1,dtype=torch.long),counts.cumsum(0)])
    total=weights.double().sum();features=field.grouping_features(scale)
    centre=(field.xyz[support.indices]*support.weights.to(field.xyz)[:,None]).sum(0)/support.weights.sum()
    result=features.new_full((len(features),),-torch.inf)
    for start in range(0,len(counts),chunk_size):
        sizes=counts[start:start+chunk_size];padded_ids=torch.zeros(len(sizes),int(sizes.max()),dtype=torch.long)
        mass=torch.zeros_like(padded_ids,dtype=torch.float32)
        for offset,size in enumerate(sizes.tolist()):
            a,b=starts[start+offset:start+offset+2].tolist();padded_ids[offset,:size]=ids[a:b];mass[offset,:size]=weights[a:b]
        ray_mass=mass.double().sum(1)
        if aggregation=="confidence_union" and bool((ray_mass>1+1e-5).any()):raise ValueError("ray confidence must be fractional registered-pixel mass")
        prior=(ray_mass/total).log().to(features)
        group,centres,_,local=moments(field,features,padded_ids.to(features.device),mass.to(features.device))
        if shared_centre:centres=centre[None].expand(len(sizes),-1)
        stats=(F.normalize(group,dim=-1),centres,features.new_zeros(len(sizes)),None if local is None else F.normalize(local,dim=-1))
        logits=logits_from_moments(field,features,scale,stats)
        if aggregation=="mixture":
            result=torch.logaddexp(result,torch.logsumexp(logits+prior,1))
        else:
            if aggregation=="confidence_union":
                probability=logits.sigmoid()*ray_mass.clamp(0,1).to(features)[None,:]
                logits=torch.logit(probability.clamp(1e-7,1-1e-7))
            result=torch.maximum(result,logits.max(1).values)
    return result


@torch.no_grad()
def query(field,packet,*,shared_centre=False,aggregation="mixture"):
    packet.validate(len(field.xyz),field.semantic.native_dim,field.local.native_dim)
    if packet.positive is None or packet.modality not in {'point_2d','point_3d'}:raise ValueError('positive rays need explicit spatial support')
    if packet.positive.components is None:return field.query(packet)
    if field.support_aggregation!='normalized_prototype' or field.relation is not None or getattr(field.evidence,'replaces_member_prior',False):
        raise ValueError('ray mixture control requires unchanged prototype energy without residual/forest')
    p=packet.positive;ids=p.indices.to(field.xyz.device);w=p.weights.to(field.xyz);valid=field.member_available[ids]&(w>0)
    if not bool(valid.any()):raise ValueError('no supported positive ray')
    support=Support(ids[valid],w[valid],p.components.to(ids.device)[valid]);scale=field.default_scale if packet.scale is None else packet.scale
    logits=mixture_logits(field,support,scale,shared_centre=shared_centre,aggregation=aggregation)
    if field.evidence is not None:logits=field.evidence.refine(logits,support,scale)
    probability=field.exclude_membership(logits.sigmoid(),packet.negative,scale,logits).masked_fill(~field.member_available,0)
    if packet.exact_positive:
        probability[support.indices]=1
        if packet.negative is not None:probability[packet.negative.indices[packet.negative.weights>0]]=0
    return QueryResult(None,probability,probability>field.threshold,support.indices,support.weights/support.weights.sum(),float(scale),
        dict(shared_member_core=True,source_access=False,positive_rays=int(support.components.unique().numel()),
        positive_aggregation={'mixture':'mass-normalized log mean exp of normalized per-ray member energies','union':'maximum per-ray member energy','confidence_union':'maximum confidence-weighted per-ray membership'}[aggregation],
        positive_centre='pooled' if shared_centre else 'per_ray',ray_subsampling=False,hard_ray_constraints=False,
        source_rgb_prompt_expansion_transferred=False,scores_calibrated=False))
