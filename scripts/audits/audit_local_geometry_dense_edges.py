"""Compare frozen carriers on all declared fusion-heldout MoGe depth views.

References remain model predictions, not ground-truth geometry. Old source
calibration may have used these views; they are held out only from fusion/new fit.
"""
import argparse
import json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scripts.repair_moge_geometry import load_views
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.audit_geometry_edge_transport import depth_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    old = base / 'geometry_only_repair_final_20260914' / a.scene
    new = base / 'local_calibrated_geometry_20260916' / a.scene
    report = json.loads((old / 'report.json').read_text())
    candidate = json.loads((new / 'report.json').read_text())
    prior = MogeGeometryPrior(report['source_report'], report['source_report_sha256'])
    frames = candidate['heldout_frames']
    prior.records = {f: prior.records[f] for f in frames}
    views = load_views(prior, 480, a.device)
    models = {
        'parent': FrozenGaussianGeometry.load(old / 'independent_scale.pt', expected_sha256=candidate['parent_geometry_sha256'], device=a.device),
        'local': FrozenGaussianGeometry.load(new / 'geometry.pt', expected_sha256=candidate['candidate_geometry_sha256'], device=a.device),
    }
    rows = []
    for view in views:
        h, w = view['depth'].shape
        entry = {'frame': view['frame'], 'variants': {}}
        for name, model in models.items():
            image, alpha, _ = rasterization(
                means=model.xyz, quats=model.quaternions, scales=model.scales,
                opacities=model.opacities.flatten(), colors=torch.zeros(len(model.xyz), 1, device=a.device),
                viewmats=torch.linalg.inv(view['pose'])[None], Ks=view['k'][None], width=w, height=h,
                near_plane=.001*prior.scale, far_plane=100*prior.scale, render_mode='RGB+ED', packed=False)
            depth = image[0, ..., -1]
            statistics = depth_stats(depth, alpha[0, ..., 0], view, prior.scale)
            valid = view['valid']
            covered = valid & (alpha[0, ..., 0] > .1) & torch.isfinite(depth) & (depth > 0)
            error = ((depth-view['depth']).abs()/view['depth'].clamp_min(1e-8))[covered]
            statistics['absrel_p90'] = float(error.quantile(.9)) if len(error) else None
            statistics['missing_pixels'] = int((valid & ~covered).sum())
            entry['variants'][name] = statistics
        rows.append(entry)
        print(view['frame'], {k: (v['coverage'], v['absrel']) for k, v in entry['variants'].items()}, flush=True)
    output = dict(scene=a.scene, reference='MoGe predictions with inherited source calibration; not geometric GT',
                  frames_requested=frames, frames_evaluated=[v['frame'] for v in views],
                  parent_sha256=candidate['parent_geometry_sha256'], candidate_sha256=candidate['candidate_geometry_sha256'],
                  source_report_sha256=report['source_report_sha256'], geometry_modified=False, rows=rows,
                  limitation='Depth discontinuities in two image axes, not an annotated semantic thin-object metric; view-mean coverage must accompany conditional errors.')
    path = Path('paper/artifacts') / f'{a.scene}_local_geometry_dense_edges_20260917.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
