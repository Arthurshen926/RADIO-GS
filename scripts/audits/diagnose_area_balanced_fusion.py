"""Source-only paired localization audit, before/after lifting and view fusion.

Caption-grounded masks are teacher pseudo-labels, not independent ground truth.
Oracle mask selection diagnoses proposal coverage and is never deployed.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', type=Path)
    p.add_argument('--weighting', choices=['conditional','contribution','pixel_partition','area_balanced'], required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(4)
    root = a.root or Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916') / a.scene
    bindings = {}
    def load(name, expected=None):
        path = root / name
        digest = sha256_file(path)
        if expected is not None:
            assert digest == expected, name
        bindings[name] = digest
        return torch.load(path, weights_only=False, map_location='cpu')
    info = json.loads((root / 'teachers.json').read_text())
    auth = json.loads((root / 'responsibility.json').read_text())
    assert sha256_file(root / 'responsibility.json') == info['authority_sha256']
    data = load('episodes.pt', json.loads((root / 'episodes.json').read_text())['sha256'])
    assert data['teacher_manifest_sha256'] == sha256_file(root / 'teachers.json')
    rec = info['teacher_files']['siglip']
    teacher = load(rec['path'], rec['sha256'])
    cap = load('caption_region_evidence.pt', json.loads((root / 'caption_region_evidence.json').read_text())['sha256'])
    assert cap['geometry_binding'] == data['geometry_binding']
    assert cap['teacher_manifest_sha256'] == data['teacher_manifest_sha256']
    trainviews = [info['frames'].index(f) for f in info['train_frames']]
    ordering = torch.cat([torch.where(cap['proposal_views'] == vi)[0] for vi in trainviews])
    assert torch.equal(ordering, torch.arange(len(cap['records']))), 'caption descriptor order mismatch'
    for ri, record in enumerate(cap['records']):
        assert record['frame'] == info['frames'][int(cap['proposal_views'][ri])]
    dev = a.device
    desc = F.normalize(teacher['descriptors'].float().to(dev), dim=-1)
    q = F.normalize(cap['text_embeddings'].float().to(dev), dim=-1)
    weights = data['region_probability'].float().to(dev)
    masks = data['masks'].flatten(1).float().to(dev)
    from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
    if a.weighting != 'conditional':
        for vi in trainviews:
            ids = torch.where(data['proposal_views'] == vi)[0]
            rec = auth['views'][vi]
            hit = load(rec['relative_path'], rec['sha256'])
            from radio_gs.training.source_area_balancing import area_balanced_pixel_votes
            source_masks=area_balanced_pixel_votes(masks[ids]) if a.weighting == 'area_balanced' else masks[ids]
            lifted, mass = source_pixel_evidence_weights(
                source_masks, hit['pixel_ids'].long().to(dev), hit['gaussian_ids'].long().to(dev),
                hit['base_weights'].float().to(dev), weights.shape[1],
                partition_overlap=a.weighting == 'pixel_partition')
            weights[ids] = weights[ids] * mass[None] if a.weighting == 'contribution' else lifted
    n = weights.shape[1]
    pixels = masks.shape[1]
    train = torch.tensor([int(v) in trainviews for v in data['proposal_views']])
    with torch.inference_mode():
        total = weights[train].T @ desc[train]
        totalmass = weights[train].sum(0)
        fullcos = F.normalize(total, dim=-1) @ q.T
        # Validate that the diagnosed full-native numerator projects to deployment.
        field = load('exact_render_query_threshold96/field.pt', json.loads((root / 'exact_render_query_threshold96/report.json').read_text())['field_sha256'])
        sd = field['state_dict']; observed = sd['observed'].to(dev)
        projected = (total / totalmass[:, None].clamp_min(1e-8)) @ sd['native_decoder.weight'].to(dev)
        error = float((projected[observed] - sd['latent'].to(dev)[observed, :64]).abs().max())
        if a.weighting == 'conditional':
            torch.testing.assert_close(projected[observed], sd['latent'].to(dev)[observed, :64], atol=2e-5, rtol=2e-4)
        del projected, field, sd
        rows = []
        for vi in trainviews:
            ids = torch.where(data['proposal_views'] == vi)[0]
            localnum = weights[ids].T @ desc[ids]
            localmass = weights[ids].sum(0)
            otherids = torch.where(train & (data['proposal_views'] != vi))[0]
            othermass = weights[otherids].sum(0)
            cosines = {
                'single_view_lift': F.normalize(localnum, dim=-1) @ q.T,
                'all_view_fusion': fullcos,
                'leave_view_out': F.normalize(total - localnum, dim=-1) @ q.T,
            }
            support = {'single_view_lift': localmass > 1e-8,
                       'all_view_fusion': totalmass > 1e-8,
                       'leave_view_out': othermass > 1e-8}
            hitrec = auth['views'][vi]
            hit = load(hitrec['relative_path'], hitrec['sha256'])
            gid, pid, w = [hit[k].to(dev) for k in ['gaussian_ids', 'pixel_ids', 'base_weights']]
            gid, pid, w = gid.long(), pid.long(), w.float()
            pixelmass = torch.zeros(pixels, device=dev).scatter_add_(0, pid, w)
            normalized = w / pixelmass[pid].clamp_min(1e-8)
            mat = torch.sparse_coo_tensor(torch.stack([pid, gid]), normalized, (pixels, n)).coalesce()
            scores, cover = {}, {}
            for name, cos in cosines.items():
                s = support[name] & observed
                cov = torch.sparse.mm(mat, s.float()[:, None])[:, 0]
                value = torch.sparse.mm(mat, ((cos + 1) / 2) * s[:, None]) / cov[:, None].clamp_min(1e-8)
                scores[name] = torch.where((cov > .5)[:, None], value, torch.zeros_like(value))
                cover[name] = cov > .5
            pixelnum = masks[ids].T @ desc[ids]
            covered2d = masks[ids].sum(0) > 0
            scores['before_geometry_pixel_mix'] = torch.where(covered2d[:, None], (F.normalize(pixelnum, dim=-1) @ q.T + 1) / 2, 0)
            cover['before_geometry_pixel_mix'] = covered2d
            capids = torch.where(cap['proposal_views'] == vi)[0].tolist()
            targets = {}
            for ri in capids:
                record = cap['records'][ri]
                if record['confidence'] < .7:
                    continue
                mask = cap['masks'][ri].flatten().to(dev).bool()
                if mask.sum() < 20:
                    continue
                for qi in record['phrase_indices']:
                    targets[qi] = targets.get(qi, torch.zeros_like(mask)) | mask
            if not targets:
                continue
            queryids = sorted(targets)
            targetmat = torch.stack([targets[qi] for qi in queryids], 1)
            elementmass = torch.zeros(n, device=dev).scatter_add_(0, gid, normalized)
            lifted = torch.sparse.mm(mat.transpose(0, 1), targetmat.float()) / elementmass[:, None].clamp_min(1e-8)
            cycle = torch.sparse.mm(mat, lifted)
            regioncos = desc[ids] @ q.T
            common = torch.stack(list(cover.values())).all(0)
            for col, qi in enumerate(queryids):
                truth = targets[qi]
                intersections = (masks[ids].bool() & truth).sum(1)
                unions = (masks[ids].bool() | truth).sum(1)
                ious = intersections / unions.clamp_min(1)
                best_semantic = int(regioncos[:, qi].argmax())
                record = dict(frame=info['frames'][vi], phrase=cap['phrases'][qi], query_index=qi,
                              positive_pixels=int(truth.sum()), common_positive_support=float(common[truth].float().mean()), oracle_region_iou=float(ious.max()),
                              semantic_selected_region_iou=float(ious[best_semantic]),
                              selected_region=int(ids[best_semantic]),
                              mask_geometry_cycle_iou=float(((cycle[:, col] > .5) & truth).sum() / ((cycle[:, col] > .5) | truth).sum().clamp_min(1)))
                for name, score in scores.items():
                    record[name] = dict(average_precision=binary_ranking_metrics(score[:, qi], truth)['average_precision'],
                                        positive_support=float(cover[name][truth].float().mean()),
                                        common_support_ap=binary_ranking_metrics(torch.where(common, score[:, qi], 0), truth)['average_precision'])
                rows.append(record)
            print(a.scene, info['frames'][vi], 'queries', len(queryids), flush=True)
            del localnum, cosines, scores, pixelnum, lifted, cycle
        macro = {k: sum(r[k] for r in rows) / len(rows) for k in ['oracle_region_iou', 'semantic_selected_region_iou', 'mask_geometry_cycle_iou', 'common_positive_support']}
        for name in ['before_geometry_pixel_mix', 'single_view_lift', 'all_view_fusion', 'leave_view_out']:
            macro[name] = {k: sum(r[name][k] for r in rows) / len(rows) for k in ['average_precision', 'positive_support', 'common_support_ap']}
    result = dict(schema='pixel_evidence_fusion_diagnostic.v1', weighting=a.weighting, area_implementation_sha256=sha256_file('radio_gs/training/source_area_balancing.py'), weighting_implementation_sha256=sha256_file('radio_gs/training/source_pixel_evidence.py'), scene=a.scene, source_only=True,
                  benchmark_data_used=False, diagnostic_only=True, teacher_masks_are_not_gt=True,
                  no_model_updates=True, original_inventory_reproduced_max_error=error if a.weighting == 'conditional' else None,
                  changed_anchor_max_error=error,
                  caption_descriptor_order_verified=True, observations=len(rows), macro=macro, rows=rows,
                  protocol='96x128 source caption mask unions, confidence>=0.7, area>=20; mean native cosine per supported pixel; no sigmoid; full image denominator; independent-source-view holdout excludes descriptors from that view, not source-built geometry or teacher pseudo-label errors',
                  inputs=bindings, teacher_manifest_sha256=sha256_file(root / 'teachers.json'),
                  authority_sha256=info['authority_sha256'], implementation_sha256=sha256_file(__file__))
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(a.scene, json.dumps(macro), flush=True)


if __name__ == '__main__':
    main()
