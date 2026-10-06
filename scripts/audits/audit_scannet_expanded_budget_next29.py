"""Register nested source-only view budgets using previously sealed visibility."""
import json
from pathlib import Path

import torch

from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4)
    out = Path('output/scannet_expanded_budget_next29')
    out.mkdir(exist_ok=False)
    rows = {}
    for short in ('0000', '0062', '0070', '0097', '0140', '0347', '0400', '0590'):
        scene = 'scene' + short + '_00'
        root = Path('output/scannet_visibility_audit_next28') / scene
        report = json.loads((root / 'report.json').read_text())
        assert report['source_only'] and not report['benchmark_labels_used']
        assert not report['selected_using_semantics']
        assert sha256_file(root / 'visibility.pt') == report['visibility_sha256']
        visibility = torch.load(root / 'visibility.pt', weights_only=False)
        geo_report = json.loads((Path('output/scannet_provided_mesh_track_20260918') /
                                 scene / 'geometry_report.json').read_text())
        geo = FrozenGaussianGeometry.load(geo_report['geometry'],
            expected_sha256=report['geometry_sha256'], device='cpu')
        weight = geo.scales.prod(1) * geo.opacities.flatten()
        vis = visibility['visibility'].float()
        ids = visibility['frame_ids']
        eligible = torch.tensor([i not in report['validation_views'] for i in ids])
        covered = torch.zeros(len(weight), dtype=torch.bool)
        chosen, budgets = [], {}
        for step in range(min(64, int(eligible.sum()))):
            gain = vis @ (weight * (~covered))
            gain[~eligible] = -1
            best = int(gain.argmax())
            chosen.append(ids[best])
            eligible[best] = False
            covered |= vis[best].bool()
            if step+1 in (16, 32, 64):
                budgets[str(step+1)] = dict(frames=chosen.copy(),
                    weighted_visibility=float(weight[covered].sum()/weight.sum()))
        # Same greedy objective; equivalent sums may differ at ties only.
        old16 = vis[[ids.index(i) for i in report['equal_budget_geometry_selected_frames']]].bool().any(0)
        budgets['sealed16'] = dict(frames=report['equal_budget_geometry_selected_frames'],
                                  weighted_visibility=float(weight[old16].sum()/weight.sum()))
        rows[scene] = dict(budgets=budgets, validation_frames=report['validation_views'],
                           geometry_sha256=report['geometry_sha256'],
                           visibility_sha256=report['visibility_sha256'])
        print(scene, {k: round(v['weighted_visibility']*100, 2) for k, v in budgets.items()}, flush=True)
    result = dict(source_only=True, benchmark_data_used=False, selected_using_semantics=False,
                  larger_training_budget=True, new_semantic_metrics=False, rows=rows,
                  implementation_sha256=sha256_file(__file__))
    (out / 'registration.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
