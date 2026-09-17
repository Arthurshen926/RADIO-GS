"""Source-only analytical initialization matching the validated native fusion."""
import torch
from torch.nn import functional as F
from radio_gs.training.native_query_basis import native_query_basis


def fit_source_fusion(descriptors, region_weights, seen, dimensions=64):
    if descriptors.ndim!=2 or region_weights.ndim!=2 or len(descriptors)!=len(region_weights) or seen.shape!=(region_weights.shape[1],):raise ValueError('source fusion axes differ')
    if not torch.isfinite(region_weights).all() or (region_weights<0).any():raise ValueError('invalid source region weights')
    basis,energy=native_query_basis(descriptors,dimensions);basis=basis.to(descriptors)
    compressed=F.normalize(descriptors,dim=-1)@basis.T
    mass=region_weights.sum(0);latent=region_weights.T@compressed/mass[:,None].clamp_min(1e-8)
    valid=seen.bool()&(mass>1e-8)&(latent.norm(dim=-1)>1e-8)
    return {'basis':basis,'latent':latent*valid[:,None],'valid':valid,'energy_retained':energy}


def apply_source_fusion(field,bundle,*,freeze_decoder=False):
    if field.native_decoder is None:raise ValueError('source initialization requires native decoder readout')
    basis=bundle['basis'].to(field.latent);latent=bundle['latent'].to(field.latent);valid=bundle['valid'].to(field.observed)
    semantic=field.latent[:,:field.native_decoder.in_features]
    if basis.shape!=field.native_decoder.weight.T.shape or latent.shape!=semantic.shape or valid.shape!=field.observed.shape or valid.dtype!=torch.bool:raise ValueError('initialization axes differ')
    if not torch.isfinite(basis).all() or not torch.isfinite(latent).all() or (valid&~field.observed).any():raise ValueError('invalid or unobserved initialization')
    if not torch.allclose(basis@basis.T,torch.eye(len(basis),device=basis.device),atol=1e-4):raise ValueError('source basis must be orthonormal')
    with torch.no_grad():
        semantic[valid]=latent[valid];field.native_decoder.weight.copy_(basis.T);field.native_decoder.bias.zero_()
    field.native_decoder.requires_grad_(not freeze_decoder)
    return int(valid.sum())


@torch.no_grad()
def project_source_update(latent, rows, initial, relative_radius):
    """Bound the source-supported latent update; other elements stay trainable.

    With a fixed orthonormal native decoder this also bounds decoded vector
    displacement. It does not guarantee unchanged ranks or calibrated scores.
    """
    if not 0<relative_radius<1 or initial.shape!=(len(rows),latent.shape[1]):raise ValueError('invalid source update radius or axes')
    delta=latent[rows]-initial;limit=relative_radius*initial.norm(dim=-1,keepdim=True)
    latent[rows]=initial+delta*(limit/delta.norm(dim=-1,keepdim=True).clamp_min(1e-12)).clamp_max(1)
