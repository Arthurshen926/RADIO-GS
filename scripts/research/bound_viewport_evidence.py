"""Pixels outside admitted detector viewports are unknown, never negatives."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    inputs = json.loads(Path(a.inputs).read_text())
    scene = json.loads(Path(inputs['scene']).read_text())
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    changed = None
    for frame in scene['source_frames']:
        data = torch.load(Path(inputs['detections']) / f'identity_{frame}.pt', map_location='cpu', weights_only=False)
        if changed is None:
            changed = torch.zeros(len(data['queries']), dtype=torch.long)
        for qi, record in enumerate(data['instances']):
            if not len(record['scores']):
                continue
            height, width = record['masks'].shape[-2:]
            yy, xx = torch.meshgrid((torch.arange(height) + .5) / height,
                                    (torch.arange(width) + .5) / width, indexing='ij')
            known = torch.zeros(height, width, dtype=torch.bool)
            for x0, y0, x1, y1 in record['viewports']:
                # One raster pixel beside an artificial crop edge is uncertain.
                left = x0 + (1 / width if x0 > 0 else 0)
                right = x1 - (1 / width if x1 < 1 else 0)
                top = y0 + (1 / height if y0 > 0 else 0)
                bottom = y1 - (1 / height if y1 < 1 else 0)
                known |= (xx >= left) & (xx < right) & (yy >= top) & (yy < bottom)
            if known.all():
                continue
            unknown = (~known)[None].to(record['masks'].dtype)
            old = record.get('unknown_masks')
            record['unknown_masks'] = torch.cat([old, unknown]) if old is not None else unknown
            changed[qi] += 1
        torch.save(data, out / f'identity_{frame}.pt')
    (out / 'inputs.json').write_text(json.dumps(dict(inputs, detections=str(out),
        evidence_contract='same admitted positive instances; complement outside observed viewports and crop edge band is unknown'), indent=2) + '\n')
    (out / 'bounds.json').write_text(json.dumps(dict(queries=data['queries'],
        bounded_views=changed.tolist(), source_frames=scene['source_frames'],
        positive_masks_changed=False, new_teacher_calls=0), indent=2) + '\n')


if __name__ == '__main__':
    main()
