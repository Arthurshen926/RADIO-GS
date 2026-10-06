"""Frozen geometry adapter for an explicitly supplied, unlabelled mesh track."""
import numpy as np
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry


def build_registered_mesh_carrier(vertices, faces):
    xyz = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Finite [N,3] vertices required')
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError('Integer triangle indices required')
    if not triangles.size or triangles.min() < 0 or triangles.max() >= len(xyz):
        raise ValueError('Triangle index outside vertex table')
    cross = np.cross(xyz[triangles[:, 1]]-xyz[triangles[:, 0]], xyz[triangles[:, 2]]-xyz[triangles[:, 0]])
    normals = np.zeros_like(xyz)
    areas = np.zeros(len(xyz), dtype=np.float32)
    for j in range(3):
        np.add.at(normals, triangles[:, j], cross)
        np.add.at(areas, triangles[:, j], np.linalg.norm(cross, axis=1)/6)
    keep = (np.linalg.norm(normals, axis=1) > 1e-6) & (areas > 0)
    if not keep.any():
        raise ValueError('No nondegenerate oriented surface vertices')
    model = FrozenGaussianGeometry(torch.from_numpy(xyz[keep]), torch.from_numpy(normals[keep]),
                                   tangent_sigma=.0075, normal_sigma=.001)
    sigma = torch.from_numpy(np.sqrt(areas[keep]/np.pi)).clamp(.003, .05)
    model.scales[:, :2] = sigma[:, None]
    return model, np.flatnonzero(keep)
