"""Exact affine-decoder cosine in a compact field, without a second native map."""
import torch
from torch.nn import functional as F


def affine_decoder_cosine(latent, query, weight, bias):
    """Equal normalize(D z + b) @ normalize(q).T up to floating-point error.

    A transposed query projection alone loses the native norms and bias. The
    Gram form avoids materializing an N x native_dimension persistent feature.
    Near cancellation, recompute the norm directly for numerical stability.
    """
    if latent.ndim!=2 or query.ndim!=2 or weight.ndim!=2 or bias.ndim!=1 or latent.shape[1]!=weight.shape[1] or query.shape[1]!=weight.shape[0] or len(bias)!=weight.shape[0]:
        raise ValueError('affine decoder/query axes differ')
    if any(not torch.isfinite(x).all() for x in (latent,query,weight,bias)):
        raise ValueError('nonfinite native decoder input')
    q=F.normalize(query,dim=-1);projected=q@weight
    numerator=latent@projected.T+(q@bias)[None,:]
    gram=weight.T@weight;linear=weight.T@bias
    quadratic=(latent@gram*latent).sum(-1);cross=2*(latent@linear);constant=bias.square().sum()
    squared=quadratic+cross+constant
    scale=quadratic.abs()+cross.abs()+constant.abs()
    near=squared<=32*torch.finfo(latent.dtype).eps*scale
    norms=squared.clamp_min(1e-24).sqrt()
    if near.any():
        ids=torch.where(near)[0]
        exact=torch.cat([F.linear(latent[rows],weight,bias).norm(dim=-1).clamp_min(1e-12) for rows in ids.split(4096)])
        norms=norms.index_copy(0,ids,exact)
        # The compact numerator also cancels near zero; recompute it as well.
        actual=torch.cat([F.linear(latent[rows],weight,bias)@q.T for rows in ids.split(4096)])
        numerator=numerator.index_copy(0,ids,actual)
    return (numerator/norms[:,None]).clamp(-1,1)
