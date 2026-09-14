"""Selected-only surfel support; distinct from full-scene posterior rendering.

This removes unselected occluders. It is not a Gaussian alpha renderer and
does not by itself certify an OpenGaussian benchmark protocol.
"""
import torch
from radio_gs.v4.carrier import SurfaceVoxelCarrier


def render_selected_surface_support(carrier, probability, camera, *, threshold):
    values = torch.as_tensor(probability, dtype=torch.float32).cpu()
    if values.shape != (carrier.num_elements,) or not torch.isfinite(values).all():
        raise ValueError("selected surface probability must align and be finite")
    if not 0 < threshold < 1 or bool(((values < 0) | (values > 1)).any()):
        raise ValueError("selected surface probability or threshold out of range")
    selected = values >= threshold
    if not selected.any():
        return torch.zeros(camera.height, camera.width, dtype=torch.bool)
    subset = SurfaceVoxelCarrier(carrier.centres[selected], carrier.voxel_size,
        normals=carrier.normals[selected] if carrier.normals is not None else None,
        confidence=carrier.confidence[selected], maximum_splat_radius=carrier.maximum_splat_radius,
        surface_band_voxels=carrier.surface_band_voxels,
        maximum_contributors_per_pixel=carrier.maximum_contributors_per_pixel,
        reference_raster_shape=carrier.reference_raster_shape)
    return subset.render_posterior(torch.ones(subset.num_elements), camera) > 0
