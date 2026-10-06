"""Prepare paper8 source geometry without reading semantic labels or queries.

This is provided-mesh input, never advertised as RGB-only or sensor-depth loss.
Scene IDs are from the frozen registry. Semantic evaluation is a separate job.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from radio_gs.geometry.registered_mesh_carrier import build_registered_mesh_carrier
from radio_gs.geometry.registered_pinhole import load_registered_camera_table
from radio_gs.utils.immutable_artifacts import sha256_file

SCENES = ['scene0000_00', 'scene0062_00', 'scene0070_00', 'scene0097_00',
          'scene0140_00', 'scene0347_00', 'scene0400_00', 'scene0590_00']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            if any(s in str(args[0]) for s in ('.labels.ply', 'language_features', 'instance_annotations')):
                raise RuntimeError('Semantic input forbidden during geometry preparation')
    sys.addaudithook(guard)
    a.output.mkdir(parents=True)
    rows = []
    for scene in SCENES:
        points = Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/'points3d.ply'
        authority = Path('output/scannet_registered_sources_20260918')/scene/'source_authority.json'
        cameras = load_registered_camera_table(authority)
        ply = PlyData.read(points)
        forbidden_properties = {'label', 'semantic', 'instance_id', 'class_id'}
        if forbidden_properties.intersection(ply['vertex'].data.dtype.names):
            raise ValueError('Input contains semantic attributes')
        xyz = np.stack([ply['vertex'][k] for k in ('x', 'y', 'z')], 1)
        faces = np.stack(ply['face']['vertex_indices'])
        model, ids = build_registered_mesh_carrier(xyz, faces)
        root = a.output/scene
        root.mkdir()
        geometry = root/'geometry.pt'
        torch.save(dict(schema='radio_gs.frozen_surface_gaussian_geometry.v1',
            geometry_trainable_during_semantics=False, state_dict=model.state_dict(),
            input_track='provided_unlabelled_mesh', source_vertex_ids=torch.from_numpy(ids)), geometry)
        report = dict(schema='registered_mesh_geometry_track.v1', scene=scene,
            source_only=True, semantic_labels_used=False, input_track='provided_unlabelled_mesh',
            rgb_only=False, per_frame_sensor_depth_used=False,
            geometry=str(geometry.resolve()), geometry_sha256=sha256_file(geometry),
            geometry_binding=model.fingerprint(), source_cameras=str(authority.resolve()),
            source_cameras_sha256=sha256_file(authority), frames=sorted(cameras),
            input_mesh=str(points), input_mesh_sha256=sha256_file(points),
            elements=len(model.xyz), excluded_vertices=len(xyz)-len(ids),
            tangent_sigma_rule='sqrt(barycentric_vertex_area/pi), clamped to [0.003,0.05] metres',
            normal_sigma_metres=.001, opacity=.8,
            independent_geometry_accuracy=False, official_benchmark_result=False,
            implementation_sha256=sha256_file(__file__),
            carrier_implementation_sha256=sha256_file('radio_gs/geometry/registered_mesh_carrier.py'))
        (root/'geometry_report.json').write_text(json.dumps(report, indent=2)+'\n')
        rows.append(dict(scene=scene, report=str(root/'geometry_report.json'), sha256=sha256_file(root/'geometry_report.json')))
        print(scene, len(model.xyz), flush=True)
    (a.output/'manifest.json').write_text(json.dumps(dict(contract='scannet-ovs-paper8-v1',
        geometry_only=True, evaluation_complete=False, scenes=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
