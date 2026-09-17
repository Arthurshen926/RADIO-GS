"""Bind actual element directions to a frozen native query projection."""
import torch
from torch.nn import functional as F


def native_region_alignment(region_element_mean, teacher_descriptor, projection):
    """Supervise the mean of unit element directions, without a free decoder.

    For any unit projected query u, u @ region_element_mean is the exact
    contribution-mean element cosine. Matching the projected teacher direction
    therefore constrains the same linear scores used before the sigmoid.
    Teacher evidence and the fixed projection are never optimization variables.
    """
    if region_element_mean.ndim!=2 or teacher_descriptor.ndim!=2 or projection.ndim!=2:
        raise ValueError('alignment tensors must be matrices')
    if region_element_mean.shape!=(len(teacher_descriptor),projection.shape[0]) or teacher_descriptor.shape[1]!=projection.shape[1]:
        raise ValueError('native alignment axes differ')
    if any(not torch.isfinite(x).all() for x in (region_element_mean,teacher_descriptor,projection)):
        raise ValueError('nonfinite native alignment evidence')
    target=F.linear(teacher_descriptor.detach(),projection.detach())
    valid=target.norm(dim=-1)>1e-8
    if not valid.any():raise ValueError('native teacher projects to zero')
    target=F.normalize(target[valid],dim=-1)
    # Summed squared vector error retains its scale when latent dimension varies.
    return (region_element_mean[valid]-target).square().sum(-1).mean()


def native_element_alignment(element_directions, teacher_directions):
    """One realizable target per frozen element; no independent region targets."""
    if element_directions.ndim!=2 or element_directions.shape!=teacher_directions.shape or not len(element_directions):
        raise ValueError('native element alignment axes differ or are empty')
    if not torch.isfinite(element_directions).all() or not torch.isfinite(teacher_directions).all():
        raise ValueError('nonfinite element teacher')
    if (teacher_directions.norm(dim=-1)<=1e-8).any():raise ValueError('zero element teacher direction')
    return (F.normalize(element_directions,dim=-1)-F.normalize(teacher_directions.detach(),dim=-1)).square().sum(-1).mean()


@torch.no_grad()
def project_native_direction_cone(values, targets, minimum_cosine):
    """Project directions into a teacher cone, retaining nonzero vector norms."""
    if values.ndim!=2 or values.shape!=targets.shape or not 0<minimum_cosine<=1:
        raise ValueError('invalid native direction constraint')
    if not torch.isfinite(values).all() or not torch.isfinite(targets).all() or (targets.norm(dim=-1)<=1e-8).any():
        raise ValueError('invalid direction evidence')
    target=F.normalize(targets,dim=-1);norm=values.norm(dim=-1,keepdim=True);direction=F.normalize(values,dim=-1)
    cosine=(direction*target).sum(-1,keepdim=True);violated=(cosine<minimum_cosine)|(norm<=1e-8)
    orthogonal=direction-cosine*target;orthogonal_norm=orthogonal.norm(dim=-1,keepdim=True)
    corrected=minimum_cosine*target+(1-minimum_cosine**2)**.5*F.normalize(orthogonal,dim=-1)
    # At the antipode or at zero there is no unique tangent: the target is safe.
    corrected=torch.where(orthogonal_norm>1e-7,corrected,target)
    return torch.where(violated,corrected*torch.where(norm>1e-8,norm,torch.full_like(norm,1e-3)),values)
