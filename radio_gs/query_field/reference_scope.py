"""Compose field-owned parts using an explicitly supplied current reference mask.

No RGB, source observations, tracks, target camera or target labels enter this
conditional decoder. This is a separate input contract from sparse scribbles.
"""
import numpy as np
import torch
from .contracts import QueryResult


def shape_union(masks, reference):
    """Deterministic strict ascent in reference soft-IoU; not global optimality."""
    masks=np.asarray(masks,dtype=bool);reference=np.asarray(reference,dtype=np.float64)
    if masks.ndim!=3 or reference.shape!=masks.shape[1:] or not len(masks):
        raise ValueError('unaligned regional reference rasters')
    if not np.isfinite(reference).all() or np.any((reference<0)|(reference>1)) or reference.sum()<=0:
        raise ValueError('reference must be a nonempty fractional foreground mask')
    current=np.zeros(reference.shape,bool);ids=[];trace=[];mass=float(reference.sum())
    intersection=0.;area=0
    while True:
        new=masks & ~current[None]
        added=new.sum(axis=(1,2));inside=np.einsum('rhw,hw->r',new,reference,optimize=True)
        score=(intersection+inside)/np.maximum(area+added+mass-intersection-inside,1e-12)
        score[added==0]=-np.inf
        if ids:score[ids]=-np.inf
        best=int(score.argmax());old=intersection/max(area+mass-intersection,1e-12)
        if score[best]<=old:break
        ids.append(best);current |= masks[best]
        intersection=float(reference[current].sum());area=int(current.sum())
        trace.append(intersection/(area+mass-intersection))
    return ids,current,trace


def visible_region_scores(operator, observed, membership):
    """Full-scene visibility stays fixed while candidate membership is read out.

    Unobserved foreground rays abstain instead of being renormalized to certainty.
    ``operator`` contains normalized full-scene transmittance contributions.
    """
    mass=torch.sparse.mm(operator,observed.to(operator.dtype)[:,None])
    scores=torch.sparse.mm(operator,membership.to(operator.dtype))/mass.clamp_min(1e-8)
    return torch.where(mass>.5,scores,0.)


def reference_member_evidence(operator, observed, reference, prior, *, readout="fraction"):
    """Condition a composed set on its visible footprint in the current mask.

    This is a visibility-weighted foreground fraction, not an instance posterior
    or a rendered-mask inverse solver. No evidence preserves the prior. A tie
    also preserves the prior selection; background-majority evidence can remove
    members, but this operation cannot invent new parts outside the composition.
    """
    if readout not in {"fraction","admission"}:raise ValueError('unknown reference member readout')
    reference=torch.as_tensor(reference,device=operator.device,dtype=operator.dtype).flatten()
    if reference.numel()!=operator.shape[0] or prior.shape!=(operator.shape[1],) or observed.shape!=prior.shape:
        raise ValueError('unaligned reference member evidence')
    if not bool(torch.isfinite(reference).all()) or bool(((reference<0)|(reference>1)).any()):
        raise ValueError('reference foreground must be finite and in [0,1]')
    observed_mass=torch.sparse.mm(operator,observed.to(operator.dtype)[:,None]).flatten()
    valid=observed_mass>.5
    # Use exactly the observed-support normalization of the reference readout.
    weight=torch.where(valid,observed_mass.clamp_min(1e-8).reciprocal(),0.)
    evidence=torch.sparse.mm(operator.transpose(0,1),torch.stack([weight*reference,weight],dim=1))
    positive,mass=evidence.unbind(1)
    visible=mass>0
    fraction=positive/torch.where(visible,mass,torch.ones_like(mass))
    fraction=fraction.clamp(0,1)
    scores=prior.to(operator.dtype)*torch.where(visible,fraction,torch.ones_like(fraction))
    selected=prior & (~visible | (positive>=.5*mass))
    if readout=="admission":scores=selected.to(operator.dtype)
    return scores,selected,dict(reference_visible_members=int((prior & visible).sum()),
        reference_unseen_members=int((prior & ~visible).sum()),
        reference_rejected_members=int((prior & ~selected).sum()),
        reference_tied_members=int((prior & visible & (positive==.5*mass)).sum()),
        reference_unseen_changed=int(((selected!=prior) & ~visible).sum()),
        reference_member_readout=readout,
        reference_member_rule='visible foreground fraction; strict background majority rejects; unseen and ties retain prior',
        weak_visibility_is_not_identity=True)


@torch.no_grad()
def query(field,packet,camera,reference,world_scale=1.,*,observation="subset",member_conditioning=False,member_readout="fraction"):
    from .confidence_mixture_evidence import ConfidenceMixtureEvidence
    from .geometry import selected_alpha, projection
    from types import SimpleNamespace
    if observation not in {"subset","scene"}:raise ValueError("unknown reference observation operator")
    if member_conditioning and observation!="scene":raise ValueError("member evidence requires full-scene visibility")
    if member_readout not in {"fraction","admission"} or (member_readout!="fraction" and not member_conditioning):
        raise ValueError("member readout requires reference conditioning")
    packet.validate(len(field.xyz),field.semantic.native_dim,field.local.native_dim)
    if packet.modality!='point_2d' or packet.negative is not None or packet.exact_positive:
        raise ValueError('reference scope requires an explicit positive-only raster query')
    if not isinstance(field.evidence,ConfidenceMixtureEvidence):
        raise ValueError('reference scope requires field-owned partial region coordinates')
    reference=torch.as_tensor(reference).detach().cpu().numpy()
    if reference.ndim!=2:raise ValueError('reference must be a two-dimensional raster')
    h,w=reference.shape;b=field.evidence.matrix('positive').transpose(0,1).to_sparse_csr()
    pointer=b.crow_indices().cpu().tolist();rows=b.col_indices();masks=[]
    if observation=="subset":
        for r in range(b.shape[0]):
            selected=torch.zeros(len(field.xyz),dtype=torch.bool,device=field.xyz.device)
            selected[rows[pointer[r]:pointer[r+1]]]=True;selected &=field.member_available
            masks.append(selected_alpha(field,selected,camera,h,w,world_scale).reshape(h,w).cpu().numpy()>.5)
    else:
        geometry=SimpleNamespace(xyz=field.xyz,get_xyz=lambda:field.xyz,
            get_rotation=lambda:field.quaternions,get_scaling=lambda:field.scales,get_opacity=lambda:field.opacities)
        _,operator,_,_=projection(geometry,camera,h,w,world_scale)
        for start in range(0,b.shape[0],32):
            stop=min(start+32,b.shape[0]);membership=field.xyz.new_zeros((len(field.xyz),stop-start))
            for r in range(start,stop):membership[rows[pointer[r]:pointer[r+1]],r-start]=1
            membership*=field.member_available[:,None]
            scores=visible_region_scores(operator,field.observed,membership)
            masks.extend((scores.T.reshape(-1,h,w).cpu().numpy()>.5))
    ids,proxy,trace=shape_union(np.stack(masks),reference)
    selected=torch.zeros(len(field.xyz),dtype=torch.bool,device=field.xyz.device)
    for r in ids:selected[rows[pointer[r]:pointer[r+1]]]=True
    selected &=field.member_available
    conditioning={};membership=selected.float()
    if member_conditioning:
        membership,selected,conditioning=reference_member_evidence(operator,field.observed,reference,selected,readout=member_readout)
    actual=selected_alpha(field,selected,camera,h,w,world_scale).reshape(h,w).cpu().numpy()>.5
    intersection=float(reference[actual].sum());union=float(actual.sum()+reference.sum()-intersection)
    support=packet.positive
    return QueryResult(None,membership,selected,support.indices,support.weights/support.weights.sum(),
        float(field.default_scale if packet.scale is None else packet.scale),
        dict(field_owned_region_composition=True,source_access=False,target_camera_used_by_decoder=False,
             reference_observation=observation,
             reference_member_conditioning=member_conditioning,**conditioning,
             input_contract='explicit current reference foreground mask, positive-only',
             candidate_regions=b.shape[0],selected_regions=ids,reference_soft_iou_trace=trace,
             reference_proxy_soft_iou=trace[-1] if trace else 0.,reference_actual_soft_iou=intersection/max(union,1e-12),
             proxy_pixels_missing_in_actual=int((proxy & ~actual).sum()),
             membership_output=('visible foreground fraction inside composed set; unseen prior retained, not calibrated probability'
                                if member_conditioning and member_readout=="fraction" else
                                'binary Gaussian set membership, not calibrated probability'),
             scores_calibrated=False,global_optimum=False,learned_local_kernel_used=False))
