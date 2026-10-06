"""Combine full-image and crop instances while preserving observation bounds."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--tiles', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    inputs = json.loads(Path(a.inputs).read_text())
    scene = json.loads(Path(inputs['scene']).read_text())
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    counts = None
    for frame in scene['source_frames']:
        full = torch.load(Path(inputs['detections']) / f'identity_{frame}.pt', map_location='cpu', weights_only=False)
        tile = torch.load(Path(a.tiles) / f'identity_{frame}.pt', map_location='cpu', weights_only=False)
        if counts is None:
            counts = torch.zeros(len(full['queries']), 3, dtype=torch.long)
        for qi, (old, extra) in enumerate(zip(full['instances'], tile['instances'])):
            n, m = len(old['scores']), len(extra['scores'])
            old['viewports'] = torch.cat([torch.tensor([[0., 0., 1., 1.]]).repeat(n, 1), extra['viewports']])
            old['scores'] = torch.cat([old['scores'], extra['scores']])
            old['masks'] = torch.cat([old['masks'], extra['masks']])
            counts[qi] += torch.tensor([n > 0, m > 0, n + m > 0])
        full['provider'] = 'original_full_image_plus_four_fixed_viewports'
        torch.save(full, out / f'identity_{frame}.pt')
    manifest = dict(inputs, detections=str(out), evidence_contract='full image and literal-query viewport proposals; per-instance normalized viewports retained')
    (out / 'inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'coverage.json').write_text(json.dumps(dict(source_frames=scene['source_frames'],
        queries=[dict(query=q, original_views=int(c[0]), crop_views=int(c[1]), combined_views=int(c[2]))
                 for q, c in zip(full['queries'], counts)], original_inputs=a.inputs, crops=a.tiles,
        target_rgb_used=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
