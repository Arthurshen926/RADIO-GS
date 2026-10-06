"""Two source-weighted semantic modes with a shared native query metric.

This experimental representation replaces a single collapsed semantic mean by
two coordinate blocks inside one element vector.  It also needs one mixture
prior scalar per element.  It does not provide compatibility with old calibrated
probability or scope heads. All construction is query independent.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def native_mixture_scores(centers, mixture_weights, queries, basis, temperature):
    """Weighted log-mean-exp of exact native-decoded component cosines.

    The query is normalized in its original native space, never after projection.
    With orthonormal decoder rows, normalizing compact centers is exactly the
    native decoded feature norm. Component priors sum to one; max pooling is not
    a substitute for this operation.
    """
    if centers.ndim != 3 or centers.shape[1] != 2 or mixture_weights.shape != centers.shape[:2]:
        raise ValueError('Expected two centers and two prior masses per element')
    if basis.ndim != 2 or queries.ndim != 2 or centers.shape[2] != len(basis) or basis.shape[1] != queries.shape[1]:
        raise ValueError('Shared native-coordinate axes required')
    if not all(torch.isfinite(x).all() for x in (centers, mixture_weights, queries, basis)):
        raise ValueError('Finite native features required')
    if (mixture_weights < 0).any() or (queries.norm(dim=-1) == 0).any():
        raise ValueError('Nonnegative priors and nonzero queries required')
    mass = mixture_weights.sum(-1)
    if not torch.allclose(mass, torch.ones_like(mass), atol=1e-5, rtol=1e-5):
        raise ValueError('Mixture priors must sum to one')
    if not torch.allclose(basis @ basis.T, torch.eye(len(basis), device=basis.device), atol=1e-4, rtol=1e-4):
        raise ValueError('Shared native decoder basis must be orthonormal')
    tau = float(temperature)
    if not 0 < tau < float('inf'):
        raise ValueError('Finite positive temperature required')
    q = F.normalize(queries.float(), dim=-1) @ basis.T
    cosine = torch.einsum('nkd,qd->nkq', F.normalize(centers.float(), dim=-1), q)
    log_prior = torch.where(mixture_weights > 0, mixture_weights.log(), -torch.inf)
    # Subtract the largest cosine rather than scaling large absolute logits.
    maximum = cosine.max(dim=1).values
    return maximum + tau * torch.logsumexp((cosine-maximum[:, None])/tau + log_prior[..., None], dim=1)


@torch.no_grad()
def fit_weighted_native_two_means(descriptors, vote_elements, vote_regions, vote_weights,
                                 num_elements, *, iterations=8, chunk_size=131072):
    """Spherical two-means with full source masses and bounded gather memory.

    Descriptors are projected normalized native image features, not independently
    normalized queries. Assignment is angular; centroids retain weighted raw
    projected coordinates so their prior-weighted mean equals the old fusion.
    The global cosine-energy temperature is mean training angular residual to
    the assigned center, with a numerical floor of1e-4. No heldout evidence fits
    either modes, priors, or temperature.
    """
    if descriptors.ndim != 2 or not torch.isfinite(descriptors).all() or (descriptors.norm(dim=1)==0).any():
        raise ValueError('Finite nonzero descriptor matrix required')
    if vote_elements.ndim != 1 or vote_regions.shape != vote_elements.shape or vote_weights.shape != vote_elements.shape:
        raise ValueError('Sparse source vote arrays must align')
    if vote_elements.dtype != torch.long or vote_regions.dtype != torch.long:
        raise ValueError('Sparse vote identities must be long tensors')
    if not len(vote_elements) or num_elements <= 0 or iterations <= 0 or chunk_size <= 0:
        raise ValueError('Nonempty votes and positive fit budget required')
    if int(vote_elements.min())<0 or int(vote_elements.max())>=num_elements or int(vote_regions.min())<0 or int(vote_regions.max())>=len(descriptors):
        raise ValueError('Source vote identity out of range')
    if not torch.isfinite(vote_weights).all() or (vote_weights<=0).any():
        raise ValueError('Source vote weights must be finite and positive')
    device=descriptors.device;d=descriptors.shape[1]
    unit=F.normalize(descriptors,dim=-1)
    def chunks():
        for start in range(0,len(vote_weights),chunk_size):
            yield slice(start,min(start+chunk_size,len(vote_weights)))
    mass=torch.zeros(num_elements,device=device)
    numerator=torch.zeros((num_elements,d),device=device)
    for part in chunks():
        i,r,w=vote_elements[part],vote_regions[part],vote_weights[part]
        mass.index_add_(0,i,w);numerator.index_add_(0,i,descriptors[r]*w[:,None])
    mean=numerator/mass[:,None].clamp_min(1e-12)
    first=F.normalize(mean,dim=-1)
    max_distance=torch.full((num_elements,),-torch.inf,device=device)
    for part in chunks():
        i,r,w=vote_elements[part],vote_regions[part],vote_weights[part]
        distance=w*(1-(unit[r]*first[i]).sum(-1)).clamp_min(0)
        max_distance.scatter_reduce_(0,i,distance,reduce='amax',include_self=True)
    seed=torch.full((num_elements,),len(descriptors),device=device,dtype=torch.long)
    for part in chunks():
        i,r,w=vote_elements[part],vote_regions[part],vote_weights[part]
        distance=w*(1-(unit[r]*first[i]).sum(-1)).clamp_min(0)
        eligible=distance==max_distance[i]
        seed.scatter_reduce_(0,i[eligible],r[eligible],reduce='amin',include_self=True)
    second=descriptors[seed.clamp_max(len(descriptors)-1)]
    centers=torch.stack([mean,second],dim=1)
    centers[mass==0]=0
    assignment=torch.zeros_like(vote_elements)
    component_mass=torch.zeros((num_elements,2),device=device)
    for _ in range(iterations):
        direction=F.normalize(centers,dim=-1)
        sums=torch.zeros((num_elements*2,d),device=device)
        counts=torch.zeros(num_elements*2,device=device)
        for part in chunks():
            i,r,w=vote_elements[part],vote_regions[part],vote_weights[part]
            similarities=(direction[i]*unit[r,None]).sum(-1)
            assigned=similarities.argmax(-1);assignment[part]=assigned
            flat=i*2+assigned
            counts.index_add_(0,flat,w);sums.index_add_(0,flat,descriptors[r]*w[:,None])
        component_mass=counts.view(num_elements,2)
        updated=sums.view(num_elements,2,d)/component_mass[...,None].clamp_min(1e-12)
        centers=torch.where(component_mass[...,None]>0,updated,mean[:,None])
    mixture=component_mass/mass[:,None].clamp_min(1e-12)
    mixture[mass==0]=torch.tensor([1.,0.],device=device)
    residual=torch.zeros((),device=device,dtype=torch.float64)
    direction=F.normalize(centers,dim=-1)
    for part in chunks():
        i,r,w=vote_elements[part],vote_regions[part],vote_weights[part]
        cosine=(unit[r]*direction[i,assignment[part]]).sum(-1)
        residual+=(w.double()*(1-cosine.double()).clamp_min(0)).sum()
    raw_temperature=float(residual/vote_weights.double().sum())
    temperature=max(raw_temperature,1e-4)
    reproduction=(centers*mixture[...,None]).sum(1)
    return dict(centers=centers,mixture_weights=mixture,mass=mass,mean=mean,
        temperature=temperature,temperature_unfloored=raw_temperature,
        weighted_mean_reconstruction_max_error=float((reproduction-mean).abs().max()),
        two_nonempty_modes=int((component_mass>0).all(1).sum()),
        iterations=iterations,source_votes=len(vote_weights))
