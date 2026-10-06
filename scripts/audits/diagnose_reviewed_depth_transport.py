"""Source-only typed mask inverse warp, bypassing frozen Gaussian carriers."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_depth_contract import effective_mapping_depth_affine
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('output/expert_plan_20260918/reviewed_effective_depth_transport.json'))
    out = parser.parse_args().output
    assert not out.exists()
    review_path = Path('docs/reviews/2026-09-18-typed-multiview-scope-review.json')
    review = json.loads(review_path.read_text())
    results, bindings, unavailable = [], {}, []
    for scene in ['figurines', 'ramen']:
        tracks = [t for t in review['tracks'] if t['scene'] == scene]
        root = Path('output/v2_scope_rebuild_20260918') / scene / 'field'
        ep = root / 'caption_region_evidence.pt'
        assert all(sha256_file(ep) == t['evidence_sha256'] for t in tracks)
        evidence = torch.load(ep, weights_only=False)
        authority = json.loads((root / 'responsibility.json').read_text())
        gp = Path('output/geometry_only_repair_final_20260914') / scene / 'report.json'
        assert sha256_file(gp) == authority['metadata']['frozen_geometry_report_sha256']
        gr = json.loads(gp.read_text())
        prior = MogeGeometryPrior(gr['source_report'], gr['source_report_sha256'])
        # Reproduce the carrier's effective depth, including recovered sources
        # and its selected joint/independent scales, not just the old affine fit.
        selected = [v for v in gr['results'].values() if v.get('geometry_sha256') == authority['metadata']['frozen_geometry_sha256']]
        assert len(selected) == 1
        geometry_path = Path(selected[0]['geometry'])
        assert sha256_file(geometry_path) == selected[0]['geometry_sha256']
        geometry = torch.load(geometry_path, weights_only=False)
        scales = dict(zip(geometry['mapping_frames'], geometry['scales'].tolist()))
        recovery = {r['frame']: r for r in gr['recovered_scales']}
        bindings[str(geometry_path)] = selected[0]['geometry_sha256']
        h, w = evidence['masks'].shape[1:]
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        pixel = torch.stack([x+.5, y+.5, torch.ones_like(x)], -1).float().reshape(-1, 3)
        views = {}
        for frame in sorted({o['frame'] for t in tracks for o in t['observations']}):
            cal, rec = prior.calibration[frame], prior.records[frame]
            recovered = recovery.get(frame, {})
            if not cal['accepted'] and not recovered.get('accepted', False):
                unavailable.append(dict(scene=scene, frame=frame, reason='No admitted calibration; unknown, not negative'))
                continue
            assert not cal.get('source_geometry_override'), 'Materialized priors require explicit loading'
            assert sha256_file(rec['prediction']) == rec['prediction_sha256']
            raw = torch.load(rec['prediction'], weights_only=False)
            assert frame in scales, 'This audit requires mapping-view effective scales'
            a, b = effective_mapping_depth_affine(frame, cal, recovered,
                geometry['mapping_frames'], geometry['scales'].tolist(), prior.scale)
            depth = raw['point_map'][..., 2].float() * a + b
            valid = raw['validity'].bool() & torch.isfinite(depth) & (depth > 0)
            depth = F.interpolate(depth[None, None], size=(h, w), mode='nearest-exact')[0, 0].flatten()
            valid = F.interpolate(valid[None, None].float(), size=(h, w), mode='nearest-exact')[0, 0].flatten().bool()
            cam = authority['metadata']['cameras'][str(frame)]
            k = torch.tensor(cam['intrinsic'])
            w2c = torch.tensor(cam['world_to_camera'])
            c2w = torch.linalg.inv(w2c)
            camera = prior.cameras[frame]
            expected_k = camera.intrinsic.clone()
            expected_k[0] *= w/camera.width
            expected_k[1] *= h/camera.height
            assert torch.allclose(k, expected_k, atol=1e-4)
            assert torch.allclose(c2w, camera.camera_to_world, atol=1e-4)
            points = pixel @ torch.linalg.inv(k).T * depth[:, None]
            world = points @ c2w[:3, :3].T + c2w[:3, 3]
            views[frame] = dict(k=k, w2c=w2c, world=world, valid=valid, depth=depth)
            bindings[rec['prediction']] = rec['prediction_sha256']
        for track in tracks:
            rows = []
            for target_obs in track['observations']:
                if target_obs['frame'] not in views:
                    continue
                target = views[target_obs['frame']]
                truth = evidence['masks'][target_obs['region']].flatten().bool()
                for source_obs in track['observations']:
                    sf = source_obs['frame']
                    if sf == target_obs['frame'] or sf not in views:
                        continue
                    source = views[sf]
                    camera = target['world'] @ source['w2c'][:3, :3].T + source['w2c'][:3, 3]
                    uv = camera @ source['k'].T
                    sx = (uv[:, 0] / uv[:, 2].clamp_min(1e-8)).floor().long()
                    sy = (uv[:, 1] / uv[:, 2].clamp_min(1e-8)).floor().long()
                    inside = (camera[:, 2] > 0) & (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
                    index = sy.clamp(0, h-1)*w + sx.clamp(0, w-1)
                    valid = inside & target['valid'] & source['valid'][index]
                    residual = (camera[:, 2]-source['depth'][index]).abs()
                    consistent = valid & (residual < .08*prior.scale + .02*source['depth'][index])
                    source_mask = evidence['masks'][source_obs['region']].bool()
                    row = dict(target_frame=target_obs['frame'], source_frame=sf)
                    for name, gate in [('no_depth_gate', valid), ('depth_consistent', consistent)]:
                        pred = source_mask.flatten()[index] & gate
                        tp = (pred & truth).sum()
                        row[name] = dict(iou=float(tp/(pred | truth).sum().clamp_min(1)),
                                         recall=float(tp/truth.sum().clamp_min(1)),
                                         positive_support=float(gate[truth].float().mean()))
                    # Target-mask-assisted error tolerance diagnostic, not a shifted prediction.
                    row['source_pixel_tolerance_recall'] = {}
                    for radius in [0, 1, 2, 4, 8]:
                        dilated = F.max_pool2d(source_mask.float()[None, None], 2*radius+1, stride=1, padding=radius)[0, 0].flatten().bool()
                        row['source_pixel_tolerance_recall'][str(radius)] = float((dilated[index] & valid & truth).sum()/truth.sum().clamp_min(1))
                    selected = valid & truth
                    row['positive_depth_residual_metres_median'] = float(residual[selected].median()/prior.scale) if selected.any() else None
                    rows.append(row)
            result = dict(track=track['id'], scene=scene, pairs=rows, geometry_report_sha256=sha256_file(gp))
            result['macro_iou'] = {mode:sum(r[mode]['iou'] for r in rows)/len(rows) if rows else None for mode in ['no_depth_gate', 'depth_consistent']}
            results.append(result)
            print(track['id'], result['macro_iou'], flush=True)
    out.write_text(json.dumps(dict(schema='reviewed_depth_transport.v1', implementation_sha256=sha256_file(__file__),
        depth_contract_sha256=sha256_file('radio_gs/geometry/frozen_depth_contract.py'),
        review_sha256=sha256_file(review_path), prediction_bindings=bindings, unavailable_views=unavailable,
        limitations='Source diagnostic, target source depth used; target masks used only for scoring/tolerance diagnosis. Assistant-reviewed identity, teacher boundaries not exact GT. Depth visibility is consistency with MoGe, not independent ground truth. Different support and operator from Gaussian transfer. No learned geometry or deployed semantic model modified.', results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
