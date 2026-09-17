"""Change only tangent splat width, preserving centres, opacity and normals."""
import argparse
import copy
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

p = argparse.ArgumentParser()
p.add_argument('--geometry-report', type=Path, required=True)
p.add_argument('--output-root', type=Path, required=True)
a = p.parse_args()
report = json.loads(a.geometry_report.read_text())
g = report['results']['independent_scale']
assert sha256_file(g['geometry']) == g['geometry_sha256']
original = torch.load(g['geometry'], weights_only=False)
a.output_root.mkdir(parents=True, exist_ok=False)
for factor in [.5, .25]:
    out = a.output_root/str(factor)
    out.mkdir()
    candidate = copy.deepcopy(original)
    candidate['state_dict']['scales'][:, :2] *= factor
    assert torch.equal(candidate['state_dict']['scales'][:, 2], original['state_dict']['scales'][:, 2])
    for key in original['state_dict']:
        if key != 'scales':
            assert torch.equal(candidate['state_dict'][key], original['state_dict'][key])
    path = out/'independent_scale.pt'
    torch.save(candidate, path)
    # Old depth metrics are invalid for a changed covariance; never copy them.
    result = {k: v for k, v in report.items() if k != 'results'}
    result.update(results=dict(independent_scale=dict(geometry=str(path.resolve()), geometry_sha256=sha256_file(path))),
        parent_geometry_report_sha256=sha256_file(a.geometry_report),
        control=dict(tangent_scale_factor=factor, centres_opacity_normals_fixed=True,
            source_only=True, benchmark_data_used=False, semantic_reconstruction_required=True),
        control_implementation_sha256=sha256_file(__file__))
    (out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
