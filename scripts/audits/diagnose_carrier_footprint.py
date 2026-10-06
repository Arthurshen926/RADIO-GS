"""Physical rerender controls for surface footprint and sampling resolution."""
import json
from pathlib import Path
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.diagnose_carrier_surface_transport import measure


def main():
    torch.set_num_threads(4)
    out = Path('output/expert_plan_20260918/carrier_footprint.json')
    assert not out.exists()
    review_path = Path('docs/reviews/2026-09-18-typed-multiview-scope-review.json')
    review = json.loads(review_path.read_text())
    results, bindings, verifications = [], {}, []
    for scene in ['figurines', 'ramen']:
        tracks = [t for t in review['tracks'] if t['scene'] == scene]
        root = Path('output/v2_scope_rebuild_20260918')/scene/'field'
        ep = root/'caption_region_evidence.pt'
        assert all(sha256_file(ep) == t['evidence_sha256'] for t in tracks)
        e = torch.load(ep, weights_only=False)
        info = json.loads((root/'teachers.json').read_text())
        assert sha256_file(root/'responsibility.json') == info['authority_sha256']
        auth = json.loads((root/'responsibility.json').read_text())
        gp = Path('output/geometry_only_repair_final_20260914')/scene/'report.json'
        assert sha256_file(gp) == auth['metadata']['frozen_geometry_report_sha256']
        gr = json.loads(gp.read_text())
        spec = next(v for v in gr['results'].values() if v.get('geometry_sha256') == auth['metadata']['frozen_geometry_sha256'])
        model = FrozenGaussianGeometry.load(spec['geometry'], expected_sha256=spec['geometry_sha256'], device='cuda')
        original_scale = model.scales.clone()
        original_binding = model.fingerprint()
        p = torch.load(root/'counterfactual_extent_benefit_admission/field.pt', weights_only=False)
        field = ElementRetrievalField(**p['architecture']); field.load_state_dict(p['state_dict'])
        known = field.observed_for('image').cuda(); del field, p
        n = len(model.xyz); h, w = e['masks'].shape[1:]; views = {}
        for frame in sorted({o['frame'] for t in tracks for o in t['observations']}):
            camera = auth['metadata']['cameras'][str(frame)]; k = torch.tensor(camera['intrinsic'])
            renderer = FeatureFieldRenderer(h, w, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), near_plane=auth['metadata']['near_plane'], far_plane=auth['metadata']['far_plane']).cuda()
            vm = torch.tensor(camera['world_to_camera'], device='cuda'); mats = {}
            for name, scale, factor in [('baseline', 1., 1), ('half_sigma', .5, 1), ('double_sigma', 2., 1), ('raster_2x', 1., 2)]:
                model.scales.copy_(original_scale*scale)
                hit = rasterize_single_view_contributions(model, renderer, vm, height=h*factor, width=w*factor)
                pid, gid, weight = hit['pixel_ids'].long(), hit['gaussian_ids'].long(), hit['weights'].float()
                if name == 'baseline':
                    rec = auth['views'][info['frames'].index(frame)]; hp = root/rec['relative_path']; assert sha256_file(hp) == rec['sha256']
                    old = torch.load(hp, weights_only=False)
                    assert torch.equal(pid.cpu(), old['pixel_ids'].long()) and torch.equal(gid.cpu(), old['gaussian_ids'].long())
                    error = float((weight.cpu()-old['base_weights']).abs().max())
                    assert error < 1e-6
                    verifications.append(dict(scene=scene, frame=frame, cached_weight_max_error=error))
                if factor > 1:
                    pid = (pid//(w*factor)//factor)*w + (pid % (w*factor))//factor
                    weight = weight/(factor*factor)
                mass = torch.zeros(h*w, device='cuda').scatter_add_(0, pid, weight)
                a = torch.sparse_coo_tensor(torch.stack([pid, gid]), weight/mass[pid].clamp_min(1e-8), (h*w, n)).coalesce()
                mats[name] = (a, torch.sparse.sum(a, dim=0).to_dense())
            views[frame] = mats
            print(scene, frame, 'rerendered', flush=True)
        model.scales.copy_(original_scale); model.assert_binding(original_binding)
        for track in tracks:
            rows = []
            for observation in track['observations']:
                frame = observation['frame']; truth = e['masks'][observation['region']].flatten().bool().cuda(); row = dict(frame=frame, controls={})
                for mode, (mat, mass) in views[frame].items():
                    local = torch.sparse.mm(mat.T, truth.float()[:, None])[:, 0]/mass.clamp_min(1e-8)
                    numerator = torch.zeros(n, device='cuda'); denominator = torch.zeros_like(numerator)
                    for source in track['observations']:
                        if source['frame'] == frame: continue
                        sm, sd = views[source['frame']][mode]
                        mask = e['masks'][source['region']].flatten().float().cuda()
                        numerator += torch.sparse.mm(sm.T, mask[:, None])[:, 0]; denominator += sd
                    row['controls'][mode] = dict(same_view=measure(mat, local, (mass > 1e-6)&known, truth), leave_view_out=measure(mat, numerator/denominator.clamp_min(1e-8), (denominator > 1e-6)&known, truth))
                rows.append(row)
            result = dict(scene=scene, track=track['id'], rows=rows, macro={mode:sum(r['controls'][mode]['leave_view_out']['iou'] for r in rows)/len(rows) for mode in views[frame]})
            results.append(result); print(track['id'], result['macro'], flush=True)
        bindings[spec['geometry']] = spec['geometry_sha256']
    out.write_text(json.dumps(dict(schema='carrier_footprint.v1', implementation_sha256=sha256_file(__file__), measure_implementation_sha256=sha256_file('scripts/audits/diagnose_carrier_surface_transport.py'), review_sha256=sha256_file(review_path), bindings=bindings, baseline_verification=verifications, results=results,
        limitations='Source teacher masks with assistant-reviewed identity; parent map saw source views. No VFM feature reuse in changed geometry controls, only fixed row support and masks. Physical rerender changes covariance or samples 2x then aggregates front-to-back contributions to original grid. Neither teacher masks nor target data choose geometry parameters. These fixed controls are scored on source cases; not final retrieval gains or independent geometry GT.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
