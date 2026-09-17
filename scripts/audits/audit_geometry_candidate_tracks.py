"""Shared sparse-reference coverage/error for independently rebuilt carriers."""
import argparse
import json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.audit_geometry_all_source_tracks import stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    report = json.loads((base/'geometry_only_repair_final_20260914'/a.scene/'report.json').read_text())
    prior = MogeGeometryPrior(report['source_report'], report['source_report_sha256'])
    inputs = {r['role']:r for r in prior.report['geometry_receipt']['inputs']}
    points, _ = _read_strict_source_points(Path(inputs['colmap_sparse_points']['path']),
        {prior.cameras[f].image_id for f in prior.records}, maximum_reprojection_error=2., minimum_track_views=2)
    paths = {'parent': Path(report['results']['independent_scale']['geometry']),
        'affine_normals': base/'affine_normal_geometry_20260917'/a.scene/'geometry.pt',
        'local': base/'local_calibrated_geometry_20260916'/a.scene/'geometry.pt',
        'finer': base/'finer_geometry_20260917'/a.scene/'geometry.pt',
        'cycle_geometry': base/'feature_assisted_geometry_20260917'/a.scene/'geometry.pt',
        'cycle_feature': base/'feature_assisted_geometry_20260917'/a.scene/'feature.pt'}
    rows = []
    for name, path in paths.items():
        model = FrozenGaussianGeometry.load(path, expected_sha256=sha256_file(path), device=a.device)
        for frame in report['heldout_frames']:
            cam = prior.cameras[frame]
            w = 480; h = round(cam.height*w/cam.width)
            k = cam.intrinsic.to(a.device).clone(); k[0] *= w/cam.width; k[1] *= h/cam.height
            image, alpha, _ = rasterization(means=model.xyz, quats=model.quaternions, scales=model.scales,
                opacities=model.opacities.flatten(), colors=torch.zeros(len(model.xyz),1,device=a.device),
                viewmats=cam.world_to_camera.to(a.device)[None], Ks=k[None], width=w,height=h,
                near_plane=.001*prior.scale, far_plane=100*prior.scale, render_mode='RGB+ED', packed=False)
            depth, mass = image[0,...,-1].cpu(), alpha[0,...,0].cpu()
            for pid, xy in zip(cam.point_ids.tolist(),cam.xy):
                if pid not in points:
                    continue
                x,y = (xy*torch.tensor([w/cam.width,h/cam.height])).floor().long().tolist()
                ref = float(points[pid] @ cam.world_to_camera[2,:3] + cam.world_to_camera[2,3])
                if not (0<=x<w and 0<=y<h) or ref<=0:
                    continue
                rows.append(dict(variant=name, frame=frame, point_id=pid,
                    covered=bool(mass[y,x]>=.5), absrel=abs(float(depth[y,x])-ref)/ref))
        del model
        print('evaluated', a.scene, name, flush=True)
    cohorts = {}
    for subset in ['all', 'reserved_point_ids']:
        selected = [r for r in rows if subset=='all' or r['point_id']%5==0]
        # The reserved IDs exclude every newly fitted local calibration, but
        # inherited legacy calibration may have seen them: not an independent test.
        cohorts[subset] = {}
        for name in paths:
            rr = [r for r in selected if r['variant']==name]
            covered = [r['absrel'] for r in rr if r['covered']]
            cohorts[subset][name] = dict(observations=len(rr), coverage=len(covered)/len(rr), absrel=stats(covered),
                accurate_5percent_coverage=sum(r['covered'] and r['absrel']<=.05 for r in rr)/len(rr),
                capped_error_missing_is_one=sum(min(r['absrel'],1.) if r['covered'] else 1. for r in rr)/len(rr))
    output = Path('paper/artifacts')/f'{a.scene}_geometry_candidate_tracks_20260917.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(dict(scene=a.scene,width=480,source_only=True,benchmark_data_used=False,
        geometries={name:dict(path=str(p),sha256=sha256_file(p)) for name,p in paths.items()},
        cohorts=cohorts, rows=rows, sparse_reference_sha256=inputs['colmap_sparse_points']['sha256'],
        limitation='Sparse source SfM reference; reserved IDs are excluded from new local fits, not inherited calibration. No semantic GT or dense geometry GT.'),indent=2)+'\n')
    print(json.dumps(cohorts,indent=2),flush=True)


if __name__=='__main__':
    main()
