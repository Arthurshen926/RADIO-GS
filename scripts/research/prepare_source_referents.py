"""Enumerate every original/tiled admitted candidate, reusing identical inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser(); p.add_argument('--output', default='output/source_referent_verification')
a = p.parse_args(); torch.set_num_threads(2)
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
jobs, mappings = {}, {}
for scene in ['figurines', 'ramen', 'teatime', 'waldo_kitchen']:
    cfg = json.loads(Path(f'configs/consolidated/{scene}.scene.json').read_text())
    rgbs = {int(r['image_id'].split('_')[-1]): r['path'] for r in json.loads(Path(cfg['source_rgb_authority']).read_text())['images']}
    for cohort, path in [('original', Path('output/conditioned_identity/full_source')/scene/'identity_evidence/inputs.json'),
                         ('tiled', Path('output/tiled_source_evidence')/scene/'admitted/inputs.json')]:
        inputs = json.loads(path.read_text()); links = {}
        for frame in cfg['source_frames']:
            source = Path(inputs['detections'])/f'identity_{frame}.pt'
            data = torch.load(source, map_location='cpu', weights_only=False)
            rows = []
            for qi, record in enumerate(data['instances']):
                keys = []
                for mi, mask in enumerate(record['masks']):
                    binary = mask.float().numpy() > .5
                    key = f'{scene}:{frame}:{qi}:' + hashlib.sha256(binary.tobytes()).hexdigest()
                    keys.append(key if binary.any() else None)
                    if binary.any() and key not in jobs:
                        jobs[key] = dict(key=key, scene=scene, frame=frame, query_index=qi, query=data['queries'][qi],
                            instance_index=mi, evidence=str(source), source_rgb=rgbs[frame])
                rows.append(keys)
            links[str(frame)] = rows
        mappings[f'{cohort}:{scene}'] = dict(inputs=str(path), instances=links)
ordered = sorted(jobs.values(), key=lambda x: (x['scene'], x['frame'], x['query_index'], x['key']))
(out/'manifest.json').write_text(json.dumps(dict(jobs=ordered, mappings=mappings, query_name_routing=False,
    candidate_cap=None, source_frame_cap=None, target_rgb_or_gt_used=False), indent=2)+'\n')
print(len(ordered), 'unique nonempty source identity crops', flush=True)
