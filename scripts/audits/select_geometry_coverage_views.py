"""Select extra mapping views from a hash-bound, geometry-only authority."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.geometry.coverage_selection import select_coverage_views
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--authority', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--initial', type=int, nargs='+', required=True)
    p.add_argument('--exclude', type=int, nargs='+', required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha256_file(a.authority) != a.expected_sha256:
        raise ValueError('authority identity')
    authority = json.loads(a.authority.read_text())
    visibility = {}
    for frame, record in zip(authority['frame_indices'], authority['views']):
        if frame in a.exclude:
            continue
        path = a.authority.parent / record['relative_path']
        if sha256_file(path) != record['sha256']:
            raise ValueError('view identity')
        hit = torch.load(path, weights_only=False, map_location='cpu')
        pid, gid, weight = hit['pixel_ids'].long(), hit['gaussian_ids'].long(), hit['base_weights'].float()
        mass = torch.zeros(authority['metadata']['feature_height'] * authority['metadata']['feature_width']).scatter_add_(0, pid, weight)
        rows = torch.zeros(authority['num_gaussians']).scatter_add_(0, gid, weight / mass[pid].clamp_min(1e-8))
        visibility[frame] = rows > 1e-6
    result = select_coverage_views(visibility, a.initial, a.exclude, a.budget)
    result.update(authority_sha256=a.expected_sha256, initial_frames=a.initial,
                  excluded_frames=a.exclude, budget=a.budget, semantic_inputs_opened=False,
                  implementation_sha256=sha256_file(__file__))
    a.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
