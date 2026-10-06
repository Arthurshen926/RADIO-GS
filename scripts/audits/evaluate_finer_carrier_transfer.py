"""Fresh source-mask contribution transport for each rebuilt geometry.

Never transfers row-indexed features/support from the old field. All carriers
use all geometry rows and the same reviewed source observations/masks.
"""
import argparse
import json
from pathlib import Path
import torch
from functools import partial
from unittest.mock import patch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.diagnose_carrier_surface_transport import measure


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', required=True)
    p.add_argument('--raster-mode', choices=['classic', 'antialiased'], default='classic')
    p.add_argument('--eps2d', type=float, default=.3)
    p.add_argument('--output-name', default='reviewed_transfer.json')
    p.add_argument('--include-opacity', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    root = Path('output/finer_surface_carrier_20260918')/args.scene
    out = root/args.output_name; assert not out.exists()
    report = json.loads((root/'report.json').read_text()); assert report['complete']
    review_path = Path('docs/reviews/2026-09-18-typed-multiview-scope-review.json')
    tracks = [t for t in json.loads(review_path.read_text())['tracks'] if t['scene'] == args.scene]
    oldroot = Path('output/v2_scope_rebuild_20260918')/args.scene/'field'
    ep = oldroot/'caption_region_evidence.pt'; assert all(sha256_file(ep) == t['evidence_sha256'] for t in tracks)
    e = torch.load(ep, weights_only=False)
    info = json.loads((oldroot/'teachers.json').read_text())
    assert sha256_file(oldroot/'responsibility.json') == info['authority_sha256']
    auth = json.loads((oldroot/'responsibility.json').read_text())
    gp = Path('output/geometry_only_repair_final_20260914')/args.scene/'report.json'
    assert sha256_file(gp) == report['parent_report_sha256']
    parent = json.loads(gp.read_text())
    variants = {'parent':parent['results']['independent_scale'], **report['results']}
    if args.include_opacity:
        op = Path('output/depth_opacity_repair_20260918')/args.scene/'report.json'
        opacity = json.loads(op.read_text())
        assert opacity['parent_geometry_sha256'] == variants['parent']['geometry_sha256']
        variants = {'parent':variants['parent'], 'depth_opacity':opacity}
    results = {}
    for name, spec in variants.items():
        model = FrozenGaussianGeometry.load(spec['geometry'], expected_sha256=spec['geometry_sha256'], device='cuda')
        n = len(model.xyz); h, w = e['masks'].shape[1:]; views = {}; rows = []
        for frame in sorted({o['frame'] for t in tracks for o in t['observations']}):
            cam = auth['metadata']['cameras'][str(frame)]; k = torch.tensor(cam['intrinsic'])
            renderer = FeatureFieldRenderer(h, w, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), near_plane=auth['metadata']['near_plane'], far_plane=auth['metadata']['far_plane']).cuda()
            # Isolated audit override. Production compositor and defaults stay
            # untouched; the exact-hit path imports this function at call time.
            import gsplat
            raster = partial(gsplat.rasterization, rasterize_mode=args.raster_mode, eps2d=args.eps2d)
            with patch('gsplat.rasterization', raster):
                hit = rasterize_single_view_contributions(model, renderer, torch.tensor(cam['world_to_camera'], device='cuda'), height=h, width=w)
            assert float((hit['accumulated_alpha']-hit['rendered_alpha']).abs().max()) < 2e-4
            pid, gid, weight = hit['pixel_ids'].long(), hit['gaussian_ids'].long(), hit['weights'].float()
            mass = torch.zeros(h*w, device='cuda').scatter_add_(0, pid, weight)
            mat = torch.sparse_coo_tensor(torch.stack([pid, gid]), weight/mass[pid].clamp_min(1e-8), (h*w, n)).coalesce()
            views[frame] = (mat, torch.sparse.sum(mat, dim=0).to_dense())
        for track in tracks:
            observations = []
            for obs in track['observations']:
                frame = obs['frame']; target = e['masks'][obs['region']].flatten().bool().cuda(); mat, mass = views[frame]
                local = torch.sparse.mm(mat.T, target.float()[:, None])[:, 0]/mass.clamp_min(1e-8)
                num = torch.zeros(n, device='cuda'); den = torch.zeros_like(num)
                pairs = []
                for src in track['observations']:
                    if src['frame'] == frame: continue
                    sm, sd = views[src['frame']]; truth = e['masks'][src['region']].flatten().float().cuda()
                    sn = torch.sparse.mm(sm.T, truth[:, None])[:, 0]
                    num += sn; den += sd
                    pairs.append(dict(source_frame=src['frame'], **measure(mat, sn/sd.clamp_min(1e-8), sd > 1e-6, target)))
                observations.append(dict(frame=frame, same_view=measure(mat, local, mass > 1e-6, target), leave_view_out=measure(mat, num/den.clamp_min(1e-8), den > 1e-6, target), pairs=pairs))
            row = dict(track=track['id'], observations=observations, macro_iou=sum(r['leave_view_out']['iou'] for r in observations)/len(observations))
            rows.append(row); print(args.scene, name, track['id'], row['macro_iou'], flush=True)
        results[name] = dict(geometry_sha256=spec['geometry_sha256'], elements=n, tracks=rows)
        del model, views; torch.cuda.empty_cache()
    out.write_text(json.dumps(dict(schema='finer_carrier_transfer.v2', scene=args.scene, raster_mode=args.raster_mode, eps2d=args.eps2d, implementation_sha256=sha256_file(__file__), compositor_sha256=sha256_file('radio_gs/rendering/contribution_compositor.py'), measure_sha256=sha256_file('scripts/audits/diagnose_carrier_surface_transport.py'), review_sha256=sha256_file(review_path), geometry_report_sha256=sha256_file(root/'report.json'), results=results,
        limitations='Source-only teacher-mask diagnostic with assistant-reviewed identities. Geometry construction did not read masks. All carriers use fresh physical rerenders and their own rows, without old semantic support/features. Parent geometry saw source views; no independent benchmark claim.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
