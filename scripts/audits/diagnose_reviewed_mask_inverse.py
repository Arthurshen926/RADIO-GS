"""Diagnostic mask inversion on reviewed identities, never a deployable score.

Compare normalized transpose lifting with fitting the exact forward renderer.
Target masks are read only for reporting; fixed-step fitting uses other views.
The all-view arm is an explicitly labelled representability oracle.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def metrics(score, target, supported):
    pred = (score > .5) & supported
    tp = (pred & target).sum().item()
    return dict(iou=tp / max((pred | target).sum().item(), 1),
                precision=tp / max(pred.sum().item(), 1),
                recall=tp / max(target.sum().item(), 1),
                positive_support=float(supported[target].float().mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--objective', choices=['balanced', 'pixel'], default='balanced')
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    review_path = Path('docs/reviews/2026-09-18-typed-multiview-scope-review.json')
    review = json.loads(review_path.read_text())
    results = []
    for scene in ['figurines', 'ramen']:
        root = Path('output/v2_scope_rebuild_20260918') / scene / 'field'
        tracks = [t for t in review['tracks'] if t['scene'] == scene]
        ep = root / 'caption_region_evidence.pt'
        assert all(sha256_file(ep) == t['evidence_sha256'] for t in tracks)
        evidence = torch.load(ep, weights_only=False)
        info = json.loads((root / 'teachers.json').read_text())
        authority = json.loads((root / 'responsibility.json').read_text())
        assert sha256_file(root / 'responsibility.json') == info['authority_sha256']
        fp = root / 'counterfactual_extent_benefit_admission/field.pt'
        payload = torch.load(fp, weights_only=False)
        field = ElementRetrievalField(**payload['architecture'])
        field.load_state_dict(payload['state_dict'])
        known = field.observed_for('image').cuda()
        n = len(field.latent)
        del field, payload
        pixels = evidence['masks'][0].numel()
        views = {}
        for frame in sorted({o['frame'] for t in tracks for o in t['observations']}):
            rec = authority['views'][info['frames'].index(frame)]
            path = root / rec['relative_path']
            assert sha256_file(path) == rec['sha256']
            hit = torch.load(path, weights_only=False)
            pid, gid, w = (hit[k].cuda() for k in ['pixel_ids', 'gaussian_ids', 'base_weights'])
            pid, gid, w = pid.long(), gid.long(), w.float()
            mass = torch.zeros(pixels, device='cuda').scatter_add_(0, pid, w)
            a = torch.sparse_coo_tensor(torch.stack([pid, gid]), w / mass[pid].clamp_min(1e-8), (pixels, n)).coalesce()
            views[frame] = (a, torch.sparse.sum(a, dim=0).to_dense())
        for track in tracks:
            observations = track['observations']
            masks = {o['frame']: evidence['masks'][o['region']].flatten().bool().cuda() for o in observations}
            rows = []
            for target_obs in observations:
                target_frame = target_obs['frame']
                row = dict(target_frame=target_frame)
                for arm in ['leave_view_out', 'all_view_oracle']:
                    sources = [o['frame'] for o in observations if arm == 'all_view_oracle' or o['frame'] != target_frame]
                    numerator = torch.zeros(n, device='cuda')
                    mass = torch.zeros_like(numerator)
                    for frame in sources:
                        a, m = views[frame]
                        numerator += torch.sparse.mm(a.T, masks[frame].float()[:, None])[:, 0]
                        mass += m
                    support = (mass > 1e-6) & known
                    initial = numerator / mass.clamp_min(1e-8)
                    # Same support normalization as the existing transfer diagnostic.
                    matrices = {}
                    for frame in set(sources + [target_frame]):
                        a, _ = views[frame]
                        den = torch.sparse.mm(a, support.float()[:, None])[:, 0]
                        idx = a.indices()
                        vals = a.values() * support[idx[1]] / den[idx[0]].clamp_min(1e-8)
                        matrices[frame] = (torch.sparse_coo_tensor(idx, vals, a.shape).coalesce(), den > .5)
                    def render(value, frame):
                        return torch.sparse.mm(matrices[frame][0], value[:, None])[:, 0]
                    before = metrics(render(initial, target_frame), masks[target_frame], matrices[target_frame][1])
                    # Direct bounded probabilities avoid saturated sigmoid gradients.
                    value = initial.clone().detach().requires_grad_(True)
                    optimizer = torch.optim.Adam([value], lr=.05)
                    for _ in range(args.steps):
                        optimizer.zero_grad()
                        losses = []
                        for frame in sources:
                            score = render(value, frame)
                            valid = matrices[frame][1]
                            target = masks[frame]
                            pos, neg = valid & target, valid & ~target
                            if args.objective == 'pixel':
                                losses.append((score[valid] - target[valid].float()).square().mean())
                            else:
                                losses.append(((score[pos] - 1).square().mean() if pos.any() else score.sum()*0)
                                              + (score[neg].square().mean() if neg.any() else score.sum()*0))
                        torch.stack(losses).mean().backward()
                        optimizer.step()
                        with torch.no_grad():
                            value.clamp_(0, 1)
                    after = metrics(render(value.detach(), target_frame), masks[target_frame], matrices[target_frame][1])
                    row[arm] = dict(source_frames=sources, transpose_lift=before,
                                    fitted_forward=after,
                                    source_fit_iou=sum(metrics(render(value.detach(), f), masks[f], matrices[f][1])['iou'] for f in sources)/len(sources))
                rows.append(row)
            result = dict(track=track['id'], scene=scene, field_sha256=sha256_file(fp), observations=rows)
            result['macro'] = {arm: {stage: sum(r[arm][stage]['iou'] for r in rows)/len(rows) for stage in ['transpose_lift', 'fitted_forward']} for arm in ['leave_view_out', 'all_view_oracle']}
            results.append(result)
            print(track['id'], result['macro'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(schema='reviewed_mask_inverse.v2', implementation_sha256=sha256_file(__file__), review_sha256=sha256_file(review_path), steps=args.steps, objective=args.objective,
        training_deployment=False, benchmark_data_used=False,
        limitations='Assistant-reviewed source identities with teacher mask boundaries. Parent geometry saw source views. All-view arm consumes evaluated source masks: representability oracle, never a generalization result. LVO excludes target mask from optimization, but uses target renderer only to report. No threshold or step selection on targets. Balanced squared pixel error differs from transpose averaging; improvements do not separately identify optimizer and objective effects.', results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
