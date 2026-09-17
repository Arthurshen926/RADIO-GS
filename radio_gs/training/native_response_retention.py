"""Retain native query responses at the actual region element aggregation level."""
import torch
from torch.nn import functional as F
from radio_gs.models.native_decoder_readout import affine_decoder_cosine


def region_element_weights(matrix, support, mask, observed):
    region=mask.bool()&(support>.5)
    if not region.any():return None
    pid,gid=matrix.indices();selected=region[pid]&observed[gid]
    rows,inverse=torch.unique(gid[selected],return_inverse=True)
    weight=torch.zeros(len(rows),device=matrix.device,dtype=matrix.dtype)
    weight.scatter_add_(0,inverse,matrix.values()[selected]/support[pid[selected]].clamp_min(1e-8)/region.sum())
    return rows,weight


def native_response_loss(latent,decoder,queries,teacher,weights,temperature=.07):
    """Match mean element cosines, not normalized mean feature or hard labels.

    Source bank only. A teacher's region descriptor defines a region-level
    response, without asserting that every element has the same descriptor.
    Calibration is excluded so changing its slope cannot satisfy this loss.
    """
    if temperature<=0 or weights.shape!=(len(latent),) or len(weights)==0 or not torch.isfinite(weights).all() or (weights<0).any() or not torch.isclose(weights.sum(),weights.new_tensor(1.),atol=1e-5):raise ValueError('normalized region contribution weights required')
    q=F.normalize(queries.detach(),dim=-1);t=F.normalize(teacher.detach(),dim=-1)
    student=(affine_decoder_cosine(latent,q,decoder.weight,decoder.bias)*weights.detach()[:,None]).sum(0)
    target=t@q.T
    return F.mse_loss(student,target)+temperature**2*F.kl_div(F.log_softmax(student/temperature,dim=-1),F.softmax(target/temperature,dim=-1),reduction='sum')
