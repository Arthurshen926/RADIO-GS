"""Resolve the camera model belonging to the released source RGB domain."""
from pathlib import Path
import numpy as np
from radio_gs.data.lerf_dataset import _read_cameras_binary, _camera_params_to_intrinsics


def source_colmap_assets(scene_root):
    root = Path(scene_root)
    sidecar = root / 'hwf_cxcy.npy'
    if not sidecar.exists():
        return root / 'sparse' / '0', None
    values = np.asarray(np.load(sidecar, allow_pickle=False), dtype=np.float64).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError('invalid released-image intrinsics sidecar')
    h, w, fx, fy, cx, cy = values
    if min(h, w, fx, fy) <= 0 or h != round(h) or w != round(w):
        raise ValueError('invalid released-image dimensions or focal length')
    # NeX crops and undistorts RGB. Its dense sparse model includes BOTH the
    # matching pinhole cameras and transformed 2D feature observations. Merely
    # replacing K on the original distorted observations would also be wrong.
    sparse = root / 'dense' / 'sparse'
    if any(not (sparse / name).is_file() for name in ('cameras.bin', 'images.bin', 'points3D.bin')):
        raise ValueError('released undistorted RGB requires matching dense sparse model')
    cameras = _read_cameras_binary(sparse / 'cameras.bin')
    if not cameras:
        raise ValueError('empty released-image camera model')
    for c in cameras.values():
        if c['model_id'] not in (0, 1):
            raise ValueError('released RGB camera must be pinhole, not distorted')
        intr = _camera_params_to_intrinsics(c['model_id'], c['params'])
        if c['height'] != int(h) or c['width'] != int(w) or not np.allclose(intr, values[2:], rtol=1e-6, atol=1e-3):
            raise ValueError('dense camera and released-image sidecar disagree')
    return sparse, sidecar
