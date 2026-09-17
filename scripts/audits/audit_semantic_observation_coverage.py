"""Separate source visibility, region coverage and labelled supervision."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    receipt = json.loads((a.root / 'episodes.json').read_text())
    if sha256_file(a.root / 'episodes.pt') != receipt['sha256']:
        raise ValueError('episode identity')
    d = torch.load(a.root / 'episodes.pt', map_location='cpu', weights_only=False)
    if sha256_file(a.root / 'teachers.json') != d['teacher_manifest_sha256']:
        raise ValueError('teacher identity')
    info = json.loads((a.root / 'teachers.json').read_text())
    train = [info['frames'].index(f) for f in info['train_frames']]
    validation = [info['frames'].index(f) for f in info['validation_frames']]
    seen = d['seen'].bool()
    observed = seen[train].any(0)
    heldout_seen = seen[validation].any(0)
    rows = []
    for region in sorted({e['query_region'] for e in d['episodes'] if e['split'] == 'validation'}):
        support = d['region_probability'][region] > .7
        count = int(support.sum())
        rows.append({'region': region, 'frame': info['frames'][int(d['proposal_views'][region])],
                     'region_elements': count, 'training_observed_elements': int((support & observed).sum()),
                     'training_observed_fraction': float((support & observed).sum()) / count if count else None})
    result = {'schema': 'semantic_observation_coverage.v1', 'episode_sha256': receipt['sha256'],
              'benchmark_ground_truth': False, 'geometry_changed': False,
              'visibility_is_not_proof_of_feature_training': True,
              'num_elements': len(observed), 'training_visible_elements': int(observed.sum()),
              'validation_visible_elements': int(heldout_seen.sum()),
              'validation_visible_but_unobserved_in_training': int((heldout_seen & ~observed).sum()),
              'not_visible_in_selected_views': int((~seen.any(0)).sum()),
              'validation_query_regions': rows}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
