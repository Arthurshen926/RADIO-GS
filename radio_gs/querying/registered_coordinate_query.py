"""Encode coordinates only when a readout explicitly ignores appearance prefixes."""
import torch


def encode_registered_coordinates(world_xyz, minimum, extent, dimension):
    if dimension < 6 or world_xyz.shape != (2, 3) or minimum.shape != (3,) or extent.shape != (3,):
        raise ValueError('two signed registered points and 3D normalization required')
    if not torch.isfinite(world_xyz).all() or not torch.isfinite(minimum).all() or not torch.isfinite(extent).all() or not (extent > 0).all():
        raise ValueError('invalid registered coordinates')
    query = world_xyz.new_zeros((1, dimension))
    query[0, -6:] = ((world_xyz-minimum.to(world_xyz))/extent.to(world_xyz)).flatten()
    return query


def coordinate_only_prompt_architecture(architecture):
    return bool(architecture.get('scale_affinity')) and architecture.get('source_membership_code') is None
