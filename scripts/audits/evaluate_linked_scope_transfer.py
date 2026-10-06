"""Leave-view-out source diagnostic for one-hop, query-independent scope links."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.training.linked_scope_evidence import linked_phrase_regions
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--links', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert not args.output.exists()
    links = json.loads(args.links.read_text())
    ep = args.root / 'caption_region_evidence.pt'
    assert links['inputs'][str(ep)] == sha256_file(ep)
    evidence = torch.load(ep, weights_only=False)
    info = json.loads((args.root / 'teachers.json').read_text())
    authority = json.loads((args.root / 'responsibility.json').read_text())
    assert sha256_file(args.root / 'responsibility.json') == info['authority_sha256']
    assert evidence['source_only'] and not evidence['benchmark_data_used']
    records = evidence['records']
    qualified = {i for i, r in enumerate(records) if r['confidence'] >= .7 and int(evidence['masks'][i].sum()) >= 20}
    pixels = evidence['masks'][0].numel()
    n = authority['num_gaussians']
    maps = {}
    for frame in info['train_frames']:
        rec = authority['views'][info['frames'].index(frame)]
        hp = args.root / rec['relative_path']
        assert sha256_file(hp) == rec['sha256']
        hit = torch.load(hp, weights_only=False)
        pid, gid, weight = hit['pixel_ids'].long(), hit['gaussian_ids'].long(), hit['base_weights'].float()
        mass = torch.zeros(pixels).scatter_add_(0, pid, weight)
        mat = torch.sparse_coo_tensor(torch.stack([pid, gid]), weight / mass[pid].clamp_min(1e-8), (pixels, n)).coalesce()
        maps[frame] = (mat, torch.sparse.sum(mat, dim=0).to_dense())
    rows = []
    for qi, phrase in enumerate(evidence['phrases']):
        original = {i for i in qualified if qi in records[i]['phrase_indices']}
        frames = sorted({records[i]['frame'] for i in original})
        if len(frames) < 2:
            continue
        cache = {}
        def fusion(ids):
            num, mass = torch.zeros(n), torch.zeros(n)
            for source in sorted({records[i]['frame'] for i in ids}):
                members = tuple(sorted(i for i in ids if records[i]['frame'] == source))
                if members not in cache:
                    mask = evidence['masks'][list(members)].bool().any(0).flatten().float()
                    cache[members] = torch.sparse.mm(maps[source][0].T, mask[:, None])[:, 0]
                num += cache[members]
                mass += maps[source][1]
            return num / mass.clamp_min(1e-8), mass > 1e-6
        for heldout in frames:
            baseline, added = linked_phrase_regions(records, links['links'], qi, [heldout])
            baseline = set(baseline) & qualified
            augmented = baseline | (set(added) & qualified)
            target = evidence['masks'][[i for i in original if records[i]['frame'] == heldout]].bool().any(0).flatten()
            row = dict(query=qi, phrase=phrase, heldout_frame=heldout, added_regions=len(augmented-baseline))
            for name, ids in [('same_phrase', baseline), ('linked', augmented)]:
                value, known = fusion(ids)
                mat = maps[heldout][0]
                support = torch.sparse.mm(mat, known.float()[:, None])[:, 0]
                score = torch.sparse.mm(mat, (value*known)[:, None])[:, 0] / support.clamp_min(1e-8)
                pred = (support > .5) & (score > .5)
                tp = int((pred & target).sum())
                row[name] = dict(iou=tp/max(int((pred|target).sum()), 1), false_positive_pixels=int((pred & ~target).sum()), false_negative_pixels=int((~pred & target).sum()))
            rows.append(row)
        if qi % 50 == 0:
            print('query', qi, 'observations', len(rows), flush=True)
    changed = [r for r in rows if r['added_regions']]
    report = dict(schema='linked_scope_transfer.v1', source_only=True, benchmark_data_used=False, physical_identity_verified=False,
        heldout_endpoints_excluded_before_link_use=True, transitive_closure_used=False, observations=rows,
        macro={name:sum(r[name]['iou'] for r in rows)/len(rows) for name in ['same_phrase', 'linked']},
        changed_observations=len(changed), improvements=sum(r['linked']['iou'] > r['same_phrase']['iou']+1e-6 for r in changed),
        regressions=sum(r['linked']['iou'] < r['same_phrase']['iou']-1e-6 for r in changed),
        evidence_sha256=sha256_file(ep), links_sha256=sha256_file(args.links), implementation_sha256=sha256_file(__file__),
        limitations='Automatic source teacher masks, not verified object truth. Query observations held out from fusion and correspondence; parent geometry and teacher feature extraction not held out. No model refit or final benchmark claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(report['macro'], 'changed', len(changed), 'better/worse', report['improvements'], report['regressions'])


if __name__ == '__main__':
    main()
