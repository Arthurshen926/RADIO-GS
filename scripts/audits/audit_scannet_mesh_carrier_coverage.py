"""Source-only geometry control; provided mesh consistency is NOT independent GT.

Compare the existing frozen MoGe carrier and two deterministic mesh carriers
on every registered source camera, including rejected MoGe calibration views.
No query, semantic label, or benchmark mask is read; no candidate is promoted.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.geometry.mesh_visibility import rasterize_registered_mesh_depth
from radio_gs.geometry.registered_pinhole import load_registered_camera_table
from radio_gs.scripts.repair_moge_geometry import evaluate_geometry
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--points', type=Path, required=True)
    p.add_argument('--cameras', type=Path, required=True)
    p.add_argument('--moge-geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.points.name != 'points3d.ply' or a.output.exists():
        raise ValueError('Require declared unlabelled input and a new output directory')

    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            if any(s in str(args[0]) for s in ('.labels.ply', 'language_features')):
                raise RuntimeError('Semantic input forbidden')
    sys.addaudithook(guard)
    torch.set_num_threads(4)
    a.output.mkdir(parents=True)
    cameras = load_registered_camera_table(a.cameras)
    mesh = PlyData.read(a.points)
    xyz = np.stack([mesh['vertex'][k] for k in ('x', 'y', 'z')], 1).astype(np.float32)
    faces = np.stack(mesh['face']['vertex_indices']).astype(np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Triangular mesh required')
    cross = np.cross(xyz[faces[:, 1]]-xyz[faces[:, 0]], xyz[faces[:, 2]]-xyz[faces[:, 0]])
    normals = np.zeros_like(xyz)
    areas = np.zeros(len(xyz), dtype=np.float32)
    for j in range(3):
        np.add.at(normals, faces[:, j], cross)
        np.add.at(areas, faces[:, j], np.linalg.norm(cross, axis=1)/6)
    keep = (np.linalg.norm(normals, axis=1) > 1e-6) & (areas > 0)
    models = {'moge_mesh_calibrated': FrozenGaussianGeometry.load(
        a.moge_geometry, expected_sha256=sha256_file(a.moge_geometry), device=a.device)}
    for name in ('provided_mesh_fixed_7p5mm', 'provided_mesh_area_footprint'):
        model = FrozenGaussianGeometry(torch.from_numpy(xyz[keep]).to(a.device),
                                       torch.from_numpy(normals[keep]).to(a.device),
                                       tangent_sigma=.0075, normal_sigma=.001)
        if name.endswith('footprint'):
            # One Gaussian covers its vertex's barycentric surface area;
            # fixed source-only rule, no metric- or query-dependent fitting.
            sigma = torch.from_numpy(np.sqrt(areas[keep]/np.pi)).to(a.device).clamp(.003, .05)
            model.scales[:, :2] = sigma[:, None]
        models[name] = model
    import nvdiffrast.torch as dr
    context = dr.RasterizeCudaContext(device=a.device)
    vertices = torch.from_numpy(xyz).to(a.device)
    views = []
    for frame, c in sorted(cameras.items()):
        h = round(c.height*a.width/c.width)
        k = c.intrinsic.to(a.device).clone()
        k[0] *= a.width/c.width
        k[1] *= h/c.height
        w2c = c.world_to_camera.to(a.device)
        depth = rasterize_registered_mesh_depth(vertices@w2c[:3, :3].T+w2c[:3, 3],
            faces, k, h, a.width, context=context)
        views.append(dict(frame=frame, depth=depth, valid=torch.isfinite(depth) & (depth > 0),
                          pose=c.camera_to_world.to(a.device), k=k))
    result = dict(schema='scannet_source_mesh_carrier_control.v1',
        source_only=True, independent_geometry_accuracy=False, official_benchmark_result=False,
        reference='Provided unlabelled mesh; shared cameras and mesh input, self-consistency only.',
        view_count=len(views), frames=[v['frame'] for v in views],
        width=a.width, excluded_degenerate_vertices=int((~keep).sum()),
        inputs={str(path): sha256_file(path) for path in (a.points, a.cameras, a.moge_geometry)},
        implementation_sha256=sha256_file(__file__), candidates={})
    for name, model in models.items():
        with torch.no_grad():
            metrics = evaluate_geometry(model, views, 1.)
        metrics['reference'] = 'provided_unlabelled_mesh_shared_input_not_independent_GT'
        result['candidates'][name] = dict(num_elements=len(model.xyz), metrics=metrics)
        print(name, metrics, flush=True)
        if name.startswith('provided'):
            torch.save(dict(schema='radio_gs.frozen_surface_gaussian_geometry.v1',
                geometry_trainable_during_semantics=False,
                state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                input_track='provided_unlabelled_mesh', diagnostic_only=True), a.output/f'{name}.pt')
    (a.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
