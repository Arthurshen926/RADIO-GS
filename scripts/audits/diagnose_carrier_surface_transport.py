"""Separate depth contamination and footprint mixing on reviewed source tracks.

Controls use source depth but never masks to choose hits. This is a source
diagnostic, not a query-time algorithm: target source depth is required.
"""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_depth_contract import effective_mapping_depth_affine
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def measure(mat, value, support, truth):
    den = torch.sparse.mm(mat, support.float()[:, None])[:, 0]
    score = torch.sparse.mm(mat, (value*support)[:, None])[:, 0]/den.clamp_min(1e-8)
    pred = (score > .5) & (den > .5)
    tp = (pred & truth).sum().item()
    return dict(iou=tp/max((pred | truth).sum().item(), 1),
                recall=tp/max(truth.sum().item(), 1),
                precision=tp/max(pred.sum().item(), 1),
                support=float((den[truth] > .5).float().mean()))


def main():
    torch.set_num_threads(4)
    out = Path('output/expert_plan_20260918/carrier_surface_transport.json')
    assert not out.exists()
    review_path = Path('docs/reviews/2026-09-18-typed-multiview-scope-review.json')
    review = json.loads(review_path.read_text())
    results, bindings = [], {}
    for scene in ['figurines', 'ramen']:
        tracks = [t for t in review['tracks'] if t['scene'] == scene]
        root = Path('output/v2_scope_rebuild_20260918')/scene/'field'
        ep = root/'caption_region_evidence.pt'
        assert all(sha256_file(ep) == t['evidence_sha256'] for t in tracks)
        evidence = torch.load(ep, weights_only=False)
        info = json.loads((root/'teachers.json').read_text())
        ap = root/'responsibility.json'
        assert sha256_file(ap) == info['authority_sha256']
        authority = json.loads(ap.read_text())
        gp = Path('output/geometry_only_repair_final_20260914')/scene/'report.json'
        assert sha256_file(gp) == authority['metadata']['frozen_geometry_report_sha256']
        gr = json.loads(gp.read_text())
        selected = [v for v in gr['results'].values() if v.get('geometry_sha256') == authority['metadata']['frozen_geometry_sha256']]
        assert len(selected) == 1
        geometry_path = Path(selected[0]['geometry'])
        assert sha256_file(geometry_path) == selected[0]['geometry_sha256']
        geometry = torch.load(geometry_path, weights_only=False)
        xyz = geometry['state_dict']['xyz'].cuda()
        prior = MogeGeometryPrior(gr['source_report'], gr['source_report_sha256'])
        recovery = {r['frame']: r for r in gr['recovered_scales']}
        fp = root/'counterfactual_extent_benefit_admission/field.pt'
        payload = torch.load(fp, weights_only=False)
        field = ElementRetrievalField(**payload['architecture'])
        field.load_state_dict(payload['state_dict'])
        known = field.observed_for('image').cuda()
        del field, payload
        n = len(xyz)
        h, w = evidence['masks'].shape[1:]
        pixels = h*w
        views = {}
        for frame in sorted({o['frame'] for t in tracks for o in t['observations']}):
            cal, rec = prior.calibration[frame], prior.records[frame]
            a, b = effective_mapping_depth_affine(frame, cal, recovery.get(frame), geometry['mapping_frames'], geometry['scales'].tolist(), prior.scale)
            assert sha256_file(rec['prediction']) == rec['prediction_sha256']
            raw = torch.load(rec['prediction'], weights_only=False)
            depth = raw['point_map'][..., 2].float()*a+b
            valid = raw['validity'].bool() & torch.isfinite(depth) & (depth > 0)
            depth = F.interpolate(depth[None, None], size=(h, w), mode='nearest-exact')[0, 0].flatten().cuda()
            valid = F.interpolate(valid[None, None].float(), size=(h, w), mode='nearest-exact')[0, 0].flatten().bool().cuda()
            c = authority['metadata']['cameras'][str(frame)]
            w2c = torch.tensor(c['world_to_camera'], device='cuda')
            z = (xyz@w2c[:3, :3].T+w2c[:3, 3])[:, 2]
            vr = authority['views'][info['frames'].index(frame)]
            hp = root/vr['relative_path']
            assert sha256_file(hp) == vr['sha256']
            hit = torch.load(hp, weights_only=False)
            pid, gid, weight = (hit[k].cuda() for k in ['pixel_ids', 'gaussian_ids', 'base_weights'])
            pid, gid, weight = pid.long(), gid.long(), weight.float()
            residual = (z[gid]-depth[pid])/prior.scale
            tol = .03+.02*depth[pid]/prior.scale
            legal = valid[pid] & (residual.abs() <= tol)
            variants = {'baseline': weight, 'depth_consistent': weight*legal}
            peak = torch.zeros(pixels, device='cuda').scatter_reduce_(0, pid, weight, reduce='amax', include_self=True)
            variants['peak_contribution'] = weight*(weight == peak[pid])
            nearest = torch.full((pixels,), float('inf'), device='cuda').scatter_reduce_(0, pid, torch.where(legal, residual.abs(), float('inf')), reduce='amin', include_self=True)
            variants['nearest_depth'] = weight*legal*(residual.abs() <= nearest[pid]+1e-6)
            mats = {}
            for name, weights in variants.items():
                mass = torch.zeros(pixels, device='cuda').scatter_add_(0, pid, weights)
                mat = torch.sparse_coo_tensor(torch.stack([pid, gid]), weights/mass[pid].clamp_min(1e-8), (pixels, n)).coalesce()
                mats[name] = (mat, torch.sparse.sum(mat, dim=0).to_dense())
            mass = torch.zeros(pixels, device='cuda').scatter_add_(0, pid, weight)
            views[frame] = dict(mats=mats, pid=pid, weight=weight/mass[pid].clamp_min(1e-8), residual=residual, legal=legal, valid=valid)
            bindings[str(hp)] = vr['sha256']
            bindings[rec['prediction']] = rec['prediction_sha256']
        for track in tracks:
            rows = []
            for target_obs in track['observations']:
                frame = target_obs['frame']
                target = evidence['masks'][target_obs['region']].flatten().bool().cuda()
                tv = views[frame]
                positive_hits = target[tv['pid']] & tv['valid'][tv['pid']]
                mass = tv['weight'][positive_hits].sum().clamp_min(1e-8)
                contamination = {}
                for name, gate in [('off_surface', ~tv['legal']), ('foreground_3cm', tv['residual'] < -.03), ('background_3cm', tv['residual'] > .03)]:
                    contamination[name] = float(tv['weight'][positive_hits & gate].sum()/mass)
                row = dict(frame=frame, positive_contribution_depth=contamination, controls={})
                for mode in tv['mats']:
                    mat, local_mass = tv['mats'][mode]
                    local = torch.sparse.mm(mat.T, target.float()[:, None])[:, 0]/local_mass.clamp_min(1e-8)
                    num = torch.zeros(n, device='cuda'); total = torch.zeros_like(num); pairs = []
                    for source in track['observations']:
                        if source['frame'] == frame: continue
                        sm, mass = views[source['frame']]['mats'][mode]
                        truth = evidence['masks'][source['region']].flatten().float().cuda()
                        lifted = torch.sparse.mm(sm.T, truth[:, None])[:, 0]
                        num += lifted; total += mass
                        pairs.append(dict(source_frame=source['frame'], **measure(mat, lifted/mass.clamp_min(1e-8), (mass > 1e-6)&known, target)))
                    row['controls'][mode] = dict(same_view=measure(mat, local, (local_mass > 1e-6)&known, target), leave_view_out=measure(mat, num/total.clamp_min(1e-8), (total > 1e-6)&known, target), pairs=pairs)
                rows.append(row)
            result = dict(track=track['id'], scene=scene, rows=rows)
            result['macro'] = {mode:sum(r['controls'][mode]['leave_view_out']['iou'] for r in rows)/len(rows) for mode in rows[0]['controls']}
            results.append(result)
            print(track['id'], result['macro'], flush=True)
        bindings[str(geometry_path)] = selected[0]['geometry_sha256']
        bindings[str(fp)] = sha256_file(fp)
    out.write_text(json.dumps(dict(schema='carrier_surface_transport.v1', implementation_sha256=sha256_file(__file__), depth_contract_sha256=sha256_file('radio_gs/geometry/frozen_depth_contract.py'), review_sha256=sha256_file(review_path), bindings=bindings, results=results,
        limitations='Source-only diagnostic on assistant-reviewed identities and uncertain teacher boundaries. Target source depth used by depth controls. Masks only score fixed geometry controls; no geometry training, no deployed model changes. Hit filtering renormalizes surviving contributions and is not a physical rerender; coverage/support accompanies IoU. Depth discrepancy is centre depth versus source MoGe, not exact tangent-plane intersection.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
