"""Post-evaluation scope audit; teacher pixels are never training inputs here."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .storage import load_field, sha256


@torch.no_grad()
def run(scene, root, output, device='cuda'):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('preserve the existing scope audit')
    current = root/f'{scene}_native_fitted_metric_v1'
    field, manifest = load_field(current/'models/trained', device)
    if field.local_metric != 'native_cosine' or field.threshold != .5:
        raise ValueError('bound assumes the declared native metric and threshold')
    deltas = json.loads((root/f'{scene}_native_point_deltas.json').read_text())
    integrity = json.loads((root/'native_local_integrity.json').read_text())
    previous = integrity[scene+'_frozen_metric']
    temperature = float(field.log_temperature.exp().clamp(1, 50))
    bias = float(field.affinity_bias)
    rows, views = [], {}
    for item in deltas['rows']:
        row = dict(item)
        frame = item['frame']
        if frame not in views:
            views[frame] = torch.load(current/'observations'/f'view_{frame:05d}.pt', map_location='cpu', weights_only=True)
        view = views[frame]
        mask = view['masks'][item['region']].flatten()
        ids = view['best_element'][mask & (view['best_element'] >= 0)].to(device)
        ids = ids[field.observed[ids]]
        anchor = int(view['best_element'][item['pixel']]) if item['pixel'] is not None else -1
        if not len(ids) or anchor < 0 or not bool(field.observed[anchor]):
            row['status'] = 'unsupported'
        else:
            local = F.normalize(field.local.decode(ids), dim=-1)
            query = F.normalize(field.local.decode(torch.tensor([anchor], device=device)), dim=-1)[0]
            cosine = (local @ query).clamp(-1, 1)
            distance = ((field.xyz[ids]-field.xyz[anchor])/(field.default_scale*field.world_units_per_metre)).square().sum(-1)/2
            # The learned group cosine is at most one. With all current scalar
            # parameters and scope held fixed, no group feature can exceed this.
            upper = temperature*(cosine-bias)-distance
            previous_upper = previous['temperature']*(1-previous['affinity_bias'])-distance
            row.update(status='ok', foreground_pixels_with_observed_dominant=len(ids),
                mean_local_cosine=float(cosine.mean()), native_bound_excluded_fraction=float((upper <= 0).float().mean()),
                previous_spatial_bound_excluded_fraction=float((previous_upper <= 0).float().mean()),
                target_scope_m=float(view['scales_m'][item['region']]), default_scope_m=field.default_scale)
        rows.append(row)
    groups = {}
    for name, selected in [('all', rows), ('improved', [r for r in rows if r['delta'] > 0]),
                           ('regressed', [r for r in rows if r['delta'] < 0])]:
        valid = [r for r in selected if r['status'] == 'ok']
        groups[name] = dict(cases=len(selected), supported_cases=len(valid),
            mean_native_bound_excluded_fraction=sum(r['native_bound_excluded_fraction'] for r in valid)/max(1, len(valid)),
            mean_previous_bound_excluded_fraction=sum(r['previous_spatial_bound_excluded_fraction'] for r in valid)/max(1, len(valid)))
    result = dict(scene=scene, field_sha256=manifest['files']['field.pt']['sha256'],
        deltas_sha256=sha256(root/f'{scene}_native_point_deltas.json'), groups=groups, rows=rows,
        evaluation_only=True, scope='Observed dominant Gaussians of teacher-region foreground pixels, not true 3D object labels.',
        limitation='Upper bound fixes the fitted scalar parameters and default scale. It does not rule out other learned bandwidths or prove all errors are native-distance errors.')
    output.write_text(json.dumps(result, indent=2)+'\n')
    return groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(run(args.scene, args.root, args.output, args.device), indent=2))
