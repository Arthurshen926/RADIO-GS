"""Measure source-region correspondence discrimination without benchmark labels."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    root = base / 'scope_consistent_episodes_20260916' / a.scene
    features = base / 'moge_feature_teacher_20260917' / a.scene
    manifest = json.loads((features / 'manifest.json').read_text())
    info = json.loads((root / 'teachers.json').read_text())
    receipt = json.loads((root / 'episodes.json').read_text())
    assert sha256_file(root / 'episodes.pt') == receipt['sha256']
    data = torch.load(root / 'episodes.pt', weights_only=False)
    sam = torch.load(root / 'teacher_sam.pt', weights_only=False)
    masks = sam['masks'].float().flatten(1)
    views = sam['view_indices']
    banks = {}
    for record in manifest['records']:
        assert sha256_file(record['path']) == record['sha256']
        blob = torch.load(record['path'], weights_only=False)
        vi = info['frames'].index(record['frame'])
        rows = torch.where(views == vi)[0]
        for key in ('encoder', 'refiner'):
            dense = blob[key].float().flatten(1)
            if key not in banks:
                banks[key] = torch.zeros(len(masks), len(dense))
            banks[key][rows] = masks[rows] @ dense.T / masks[rows].sum(1).clamp_min(1)[:, None]
    dino = data['queries']['region_dino'].float()
    banks['dino'] = dino
    training_views = {info['frames'].index(f) for f in info['train_frames']}
    training_rows = torch.tensor([int(v) in training_views for v in views])
    for key in ('encoder', 'refiner'):
        # Fixed source-training centering probes anisotropy; no heldout fitting.
        banks[key + '_centered'] = banks[key] - banks[key][training_rows].mean(0)
    result = {}
    for key, bank in banks.items():
        bank = F.normalize(bank, dim=1)
        records = []
        unique = set()
        for e in data['episodes']:
            if e['modality'] != 'image':
                continue
            q, t = e['query_region'], e['target_region']
            if (q, t) in unique:
                continue
            unique.add((q, t))
            # Only explicitly corroborated negatives; unknown objects aren't negatives.
            negative = e['negative_regions']
            if not negative:
                continue
            positive = float(bank[q] @ bank[t])
            scores = bank[negative] @ bank[q]
            records.append(dict(split=e['split'], query=q, target=t,
                                positive_cosine=positive,
                                negative_cosine=float(scores.mean()),
                                pairwise_accuracy=float((positive > scores).float().mean()),
                                top1=bool(positive > scores.max())))
        result[key] = {split: dict(
            count=sum(r['split'] == split for r in records),
            **{metric: (sum(float(r[metric]) for r in records if r['split'] == split) /
                        sum(r['split'] == split for r in records))
               if any(r['split'] == split for r in records) else None
               for metric in ('positive_cosine', 'negative_cosine', 'pairwise_accuracy', 'top1')})
            for split in ('train', 'validation')}
        result[key]['records'] = records
    output = dict(scene=a.scene, episode_sha256=receipt['sha256'],
                  manifest_sha256=sha256_file(features / 'manifest.json'), results=result,
                  caveat='Region-mean descriptors and existing pseudo-correspondences; DINO helped admit pairs, so its advantage is not an independent teacher ranking. No pixel correspondence or GT accuracy asserted.')
    path = Path('paper/artifacts') / f'{a.scene}_moge_region_correspondence_20260917.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: {s: v[s] for s in ('train', 'validation')} for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
