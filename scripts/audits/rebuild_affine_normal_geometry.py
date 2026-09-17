"""Isolate carrier sampling/footprint from depth calibration and semantic learning."""
import argparse
import json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scripts.repair_moge_geometry import load_views, build_geometry
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.audit_geometry_edge_transport import depth_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--voxel-metres', type=float, default=.01)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    a.output.mkdir(parents=True, exist_ok=False)
    path = Path('/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914') / a.scene / 'report.json'
    report = json.loads(path.read_text())
    prior = MogeGeometryPrior(report['source_report'], report['source_report_sha256'])
    all_records = prior.records
    prior.records = {f: all_records[f] for f in report['mapping_frames']}
    views = load_views(prior, report['width'], a.device, allow_global_scale_fallback=True)
    recovered = {r['frame']: r for r in report['recovered_scales'] if r['accepted']}
    for view in views:
        if not view['local_calibration_accepted']:
            view['depth'] *= recovered[view['frame']]['scale']
    candidate, fused = build_geometry(views, torch.ones(len(views), device=a.device), prior.scale,
                                     a.voxel_metres, a.device, report['minimum_voxel_views'])
    candidate_path = a.output / 'geometry.pt'
    parent_record = report['results']['independent_scale']
    torch.save(dict(schema='radio_gs.frozen_surface_gaussian_geometry.v1',
                    state_dict={k: v.cpu() for k, v in candidate.state_dict().items()},
                    geometry_trainable_during_semantics=False, parent_geometry_sha256=parent_record['geometry_sha256'],
                    mapping_frames=report['mapping_frames'], source_authority_sha256=prior.source_authority_sha256), candidate_path)
    print('built', len(candidate.xyz), flush=True)
    del views, fused
    parent = FrozenGaussianGeometry.load(parent_record['geometry'], expected_sha256=parent_record['geometry_sha256'], device=a.device)
    prior.records = {f: all_records[f] for f in report['heldout_frames']}
    rows = []
    for view in load_views(prior, 480, a.device):
        h, w = view['depth'].shape
        row = dict(frame=view['frame'], variants={})
        for name, model in [('parent', parent), ('normal_corrected', candidate)]:
            image, alpha, _ = rasterization(means=model.xyz, quats=model.quaternions, scales=model.scales,
                opacities=model.opacities.flatten(), colors=torch.zeros(len(model.xyz), 1, device=a.device),
                viewmats=torch.linalg.inv(view['pose'])[None], Ks=view['k'][None], width=w, height=h,
                near_plane=.001*prior.scale, far_plane=100*prior.scale, render_mode='RGB+ED', packed=False)
            row['variants'][name] = depth_stats(image[0, ..., -1], alpha[0, ..., 0], view, prior.scale)
        rows.append(row)
        print(view['frame'], {k: (v['coverage'], v['absrel']) for k, v in row['variants'].items()}, flush=True)
    result = dict(schema='affine_normal_geometry_control.v1', scene=a.scene, voxel_metres=a.voxel_metres,
        source_only=True, benchmark_data_used=False, semantics_used=False, parent_elements=len(parent.xyz),
        candidate_elements=len(candidate.xyz), candidate_sha256=sha256_file(candidate_path),
        parent_sha256=parent_record['geometry_sha256'], source_report_sha256=sha256_file(path),
        implementation_sources={p: sha256_file(p) for p in [__file__, 'radio_gs/scripts/repair_moge_geometry.py', 'radio_gs/v4/geometry/tsdf_fusion.py', 'radio_gs/geometry/affine_depth_normals.py']},
        rows=rows, promoted=False, centre_max_error=float((parent.xyz-candidate.xyz).abs().max()) if parent.xyz.shape==candidate.xyz.shape else None, normal_policy='inverse-transpose affine depth warp after intrinsic ray correction', limitation='Same depth/calibration/voxel policy, corrected normals only; MoGe references are not ground truth. Fusion CUDA reductions may not be bitwise deterministic.')
    (a.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
