"""Dense language evidence on a separately indexed semantic surface carrier."""
import torch
from torch.nn import functional as F
from radio_gs.querying.source_kernel_extent import attach_kernel_native


def attach_dense_surface_field(model,payload):
    attach_kernel_native(model,payload)
    state=payload['dense_surface_field'];device=model.latent.device
    features=state['features'].to(device);mapping=state['gaussian_to_element'].to(device)
    supported=state['supported'].to(device);original=model.logits
    def logits(query,modality,rows=None):
        base=original(query,modality,rows)
        if modality!='text':return base
        q=F.normalize(query.float(),dim=-1)
        cosine=torch.cat([F.normalize(x.float(),dim=-1)@q.T for x in features.split(4096)])
        if cosine.ndim==3:cosine=cosine.amax(1)
        ids=mapping if rows is None else mapping[rows]
        valid=supported if rows is None else supported[rows]
        threshold=model.native_query_thresholds['text'](q)
        value=state['scale']*(cosine[ids]-threshold+state['offset'])
        return torch.where(valid[:,None],value,base)
    model.logits=logits
    return model
