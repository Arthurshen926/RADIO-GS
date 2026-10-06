"""Opt-in registered NVOS spatial-component query runtime.

No target RGB/GT input is accepted. This is a query-time map readout, not a
persistent field mutation. Reference SAM complement remains unknown.
"""
from __future__ import annotations
import torch

def refine_spatial_prompt_query(*, scores, field, geometry, reference_camera,
                                geometry_prior, raster_wh, positive_mask,
                                reference_sam_positive_mask):
    from scripts.audits.nvos_system_refinement_20260919_v3_evaluate import refine
    width,height=map(int,raster_wh)
    if positive_mask.dtype!=torch.bool or reference_sam_positive_mask.dtype!=torch.bool:
        raise ValueError('Registered prompt masks must be Boolean')
    if tuple(positive_mask.shape)!=(height,width) or reference_sam_positive_mask.shape!=positive_mask.shape:
        raise ValueError('Registered reference raster mismatch; no implicit resize')
    if scores.shape!=(len(geometry.xyz),1) or not torch.isfinite(scores).all():
        raise ValueError('One finite element membership score per frozen element required')
    if field.native_semantic_dim!=64 or field.latent.shape[1]!=96:
        raise ValueError('This opt-in adapter is validated only for frozen native64/structure32 NVOS maps')
    # Use the registered implementation and constants unchanged. It returns
    # temporary scores; neither field/encoder nor geometry is optimized.
    with torch.no_grad():
        return refine(scores,field.observed_for('image'),geometry,reference_camera,
                      geometry_prior,width,height,positive_mask,
                      reference_sam_positive_mask,'spatial_component',field)
