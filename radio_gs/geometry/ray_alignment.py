"""Preserve surface normals when correcting a model's pinhole intrinsics."""
import torch
from torch.nn import functional as F


def align_normals_to_intrinsics(normals, source_intrinsics, target_intrinsics):
    """For p_new=K_target^-1 K_source p_old, use n_new=A^-T n_old.

    Both intrinsic matrices must use the same image coordinate units. Merely
    re-unprojecting depth while keeping old normals changes the tangent plane.
    """
    n=torch.as_tensor(normals).float()
    source=torch.as_tensor(source_intrinsics,device=n.device).float()
    target=torch.as_tensor(target_intrinsics,device=n.device).float()
    if n.shape[-1]!=3 or source.shape!=(3,3) or target.shape!=(3,3):
        raise ValueError('invalid ray alignment shapes')
    if not all(bool(torch.isfinite(v).all()) for v in (n,source,target)):
        raise ValueError('nonfinite ray alignment inputs')
    transform=torch.linalg.solve(target,source)
    return F.normalize(n@torch.linalg.inv(transform),dim=-1)
