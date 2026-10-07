"""Post-study source-only audit of information discarded by prototype pooling."""
import argparse
import json
from pathlib import Path

import torch

from .observations import training_episode
from .storage import load_field, sha256
from .support_training import TrainingRays, sample_pixels


@torch.no_grad()
def audit(field_path, observations, output, device='cuda'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    field, manifest = load_field(field_path, device)
    root = Path(observations)
    source = json.loads((root/'observations.json').read_text())
    if manifest['provenance']['observations_sha256'] != sha256(root/'observations.json'):
        raise ValueError('source/field mismatch')
    episodes, views = [], []
    for entry in source['frames']:
        if entry['split'] != 'train':
            continue
        path = root/f"view_{entry['frame']:05d}.pt"
        if sha256(path) != entry['compiled']['sha256']:
            raise ValueError('source observations changed')
        view = torch.load(path, map_location='cpu', weights_only=True)
        view['rays'] = TrainingRays(view['operator'])
        views.append(view)
        for index in range(len(view['masks'])):
            episode = training_episode(view, index, field.observed.cpu())
            if episode is not None:
                _, negative, positive_pixels = episode
                negative_pixels = torch.where(negative & (view['best_element'] >= 0))[0]
                negative_pixels = negative_pixels[field.observed.cpu()[view['best_element'][negative_pixels]]]
                episodes.append((len(views)-1, index, positive_pixels, negative_pixels))
    generator = torch.Generator().manual_seed(20261007)
    rows = []
    for position in torch.linspace(0, len(episodes)-1, min(32, len(episodes))).round().long().tolist():
        vi, region, pixels, negatives = episodes[position]
        view = views[vi]
        first = pixels[len(pixels)//2]
        supports = dict(ray=view['rays'].support([int(first)], device),
                        pooled_positive=view['rays'].support(sample_pixels(pixels, first, generator), device))
        if len(negatives):
            supports['pooled_negative'] = view['rays'].support(sample_pixels(negatives, negatives[len(negatives)//2], generator), device)
        scale = float(view['scales_m'][region])
        features = field.grouping_features(scale)
        stats = {}
        for name, support in supports.items():
            valid = field.observed[support.indices]
            ids, weights = support.indices[valid], support.weights[valid]
            weights = weights/weights.sum()
            proto = (features[ids]*weights[:, None]).sum(0)
            centre = (field.xyz[ids]*weights[:, None]).sum(0)
            variance = ((field.xyz[ids]-centre).square().sum(1)*weights).sum()
            stats[name] = dict(resultant_norm=float(proto.norm()),
                               spatial_variance_over_twice_scope_squared=float(variance/(2*(scale*field.world_units_per_metre)**2)),
                               support_rows=len(ids))
        rows.append(dict(frame=view['frame'], region=region, scale_m=scale, supports=stats))
    summary = {}
    for kind in ['ray', 'pooled_positive', 'pooled_negative']:
        values = [row['supports'][kind]['resultant_norm'] for row in rows if kind in row['supports']]
        summary[kind] = dict(cases=len(values), mean_resultant_norm=sum(values)/len(values),
                             fraction_below_point_nine=sum(v<.9 for v in values)/len(values))
    report = dict(field_sha256=manifest['files']['field.pt']['sha256'], source_train_only=True,
                  source_observations_sha256=sha256(root/'observations.json'), eligible_episodes=len(episodes),
                  sampling='32 equally spaced eligible source episodes; median first pixel; fixed four-ray draws',
                  post_study_diagnostic=True, benchmark_queries_or_labels_read=False, summary=summary, rows=rows,
                  limitation='Resultant norm and variance are uncalibrated ambiguity proxies; no causal or accuracy claim.')
    output.write_text(json.dumps(report, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field', required=True); parser.add_argument('--observations', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); torch.set_num_threads(4)
    print(json.dumps(audit(args.field, args.observations, args.output, args.device), indent=2))
