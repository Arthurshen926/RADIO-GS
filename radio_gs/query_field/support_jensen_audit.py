"""Post-study source-only check: sigmoid(mean energy) is not mean response.

The sampled rows are dominant associations of known source interiors/exclusions,
not verified 3D membership labels. No fit, public prompt or target mask is used.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .contracts import Support
from .observations import training_episode
from .storage import load_field, sha256
from .support_energy import expected_logits
from .support_training import TrainingRays


def evenly_spaced(values, count):
    return values[torch.linspace(0, len(values)-1, min(count,len(values))).round().long()]


@torch.no_grad()
def audit(field_path, observations, output, device='cuda'):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    field, manifest = load_field(field_path, device)
    assert field.support_aggregation == 'normalized_prototype' and field.local_metric == 'none'
    assert field.spatial_metric == 'scale_normalized'
    root = Path(observations)
    source = json.loads((root/'observations.json').read_text())
    assert manifest['provenance']['observations_sha256'] == sha256(root/'observations.json')
    observed = field.observed.cpu()
    views, episodes = [], []
    for entry in source['frames']:
        if entry['split'] != 'train': continue
        path = root/f"view_{entry['frame']:05d}.pt"
        assert sha256(path) == entry['compiled']['sha256']
        view = torch.load(path, map_location='cpu', weights_only=True)
        view['rays'] = TrainingRays(view['operator'])
        views.append(view)
        for region in range(len(view['masks'])):
            episode = training_episode(view, region, observed)
            if episode is not None:
                _, negative, positives = episode
                negatives = torch.where(negative & (view['best_element'] >= 0))[0]
                negatives = negatives[observed[view['best_element'][negatives]]]
                episodes.append((len(views)-1, region, positives, negatives))
    records = []
    for position in evenly_spaced(torch.arange(len(episodes)), 32):
        vi, region, positives, negatives = episodes[position]
        view = views[vi]
        first = positives[len(positives)//2]
        support = view['rays'].support([int(first)], device)
        valid = field.observed[support.indices]
        support = Support(support.indices[valid], support.weights[valid])
        weights = support.weights/support.weights.sum()
        scale = float(view['scales_m'][region])
        features = field.grouping_features(scale)
        prototypes = F.normalize(features[support.indices], dim=-1)
        row = dict(frame=view['frame'], region=region, contributors=len(weights), scale_m=scale, groups={})
        for sign, pixels in [('source_interior',positives), ('source_exclusion',negatives)]:
            if not len(pixels): continue
            ids = evenly_spaced(view['best_element'][pixels].unique(sorted=True), 64).to(device)
            logits = field.log_temperature.exp().clamp(1,50)*(features[ids]@prototypes.T-field.affinity_bias)
            distance = (field.xyz[ids,None]-field.xyz[support.indices][None])/(scale*field.world_units_per_metre)
            logits -= .5*distance.square().sum(-1)
            # Validate against both actual implementations on the same rows.
            mean_energy = expected_logits(field, support, scale, ids)
            assert torch.allclose(mean_energy, logits@weights, atol=3e-5, rtol=1e-5)
            singleton = field.member_logits(Support(support.indices[:1], torch.ones(1,device=device)), scale, ids)
            assert torch.allclose(logits[:,0], singleton, atol=3e-5, rtol=1e-5)
            legacy = field.member_logits(support, scale, ids).sigmoid()
            expected_energy = mean_energy.sigmoid()
            response_mixture = logits.sigmoid()@weights
            row['groups'][sign] = dict(rows=len(ids),
                legacy_mean_response=float(legacy.mean()), expected_energy_mean_response=float(expected_energy.mean()),
                response_mixture_mean_response=float(response_mixture.mean()),
                mean_response_minus_response_of_mean=float((response_mixture-expected_energy).mean()),
                mean_absolute_noncommutation=float((response_mixture-expected_energy).abs().mean()),
                legacy_selected_fraction=float((legacy>.5).float().mean()),
                expected_energy_selected_fraction=float((expected_energy>.5).float().mean()),
                response_mixture_selected_fraction=float((response_mixture>.5).float().mean()))
        records.append(row)
    summary = {}
    for sign in ['source_interior','source_exclusion']:
        groups = [r['groups'][sign] for r in records if sign in r['groups']]
        summary[sign] = dict(episodes=len(groups), total_rows=sum(g['rows'] for g in groups),
            **{key:sum(g[key] for g in groups)/len(groups) for key in groups[0] if key!='rows'})
    report = dict(field_sha256=manifest['files']['field.pt']['sha256'],
        observations_sha256=sha256(root/'observations.json'), source_train_only=True,
        benchmark_queries_or_labels_read=False, post_study_diagnostic=True,
        sampling='32 uniformly spaced eligible episodes; median interior ray; <=64 uniform distinct dominant rows per known source sign',
        no_public_accuracy_or_calibration_claim=True, no_fit_or_parameter_search=True,
        limitation='Dominant row associations are uncertain and not verified 3D labels; sampled response statistics are not mask IoU.',
        summary=summary, records=records)
    output.write_text(json.dumps(report,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field', required=True); parser.add_argument('--observations', required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--device',default='cuda')
    args = parser.parse_args(); torch.set_num_threads(4)
    print(json.dumps(audit(args.field,args.observations,args.output,args.device),indent=2))
