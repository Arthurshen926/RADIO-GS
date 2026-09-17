"""Rebuild the complete retained readout on a newly reconstructed source field.

Source evidence can be reused only with identical teachers, geometry and source
frames; every field-dependent head is fitted afresh. This is a matched lineage
control, not automatic acceptance of the old readout architecture.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    old = base / 'preserved_source_inventory_20260916' / a.scene
    root = base / 'scope_consistent_episodes_20260916' / a.scene
    for name in ['teachers.json', 'responsibility.json', 'source_frames.json', 'native_query_encoder.json']:
        assert sha256_file(root/name) == sha256_file(old/name), name
    payload = torch.load(root/'scope96_protected_imagecal_balanced/field.pt', weights_only=False)
    evidence = torch.load(old/'caption_region_evidence.pt', weights_only=False)
    assert evidence['geometry_binding'] == payload['geometry_binding']
    assert evidence['teacher_manifest_sha256'] == sha256_file(root/'teachers.json')
    assert evidence['train_frames'] == json.loads((root/'teachers.json').read_text())['train_frames']
    assert evidence['source_only'] and not evidence['benchmark_data_used']
    assert 'episode_sha256' not in evidence
    for name in ['caption_region_evidence.pt', 'caption_region_evidence.json']:
        if (root/name).exists():
            raise FileExistsError(root/name)
        shutil.copy2(old/name, root/name)
    assert sha256_file(root/'caption_region_evidence.pt') == json.loads((root/'caption_region_evidence.json').read_text())['sha256']
    groups = json.loads((old/'source_membership_balanced32/report.json').read_text())
    group_path = root/'scope_membership_groups.json'
    group_path.write_text(json.dumps(dict(evidence_sha256=groups['evidence_sha256'], groups=groups['groups'],
        provenance_sha256=sha256_file(old/'source_membership_balanced32/report.json'),
        reuse_reason='Pure source masks/groups, independent of learned element vectors and episode edges.'), indent=2)+'\n')
    shutil.copytree(root/'bounded_explicit_direct_control', root/'scale_affinity_v1_direct_control')
    stages = [
        ('fit_full_source_query_threshold', []),
        ('fit_grounded_semantic_transfer', ['--group-report', 'scope_membership_groups.json']),
        ('train_source_region_extents', []),
        ('train_rendered_extent', ['--residual-dim', '128', '--steps', '800']),
        ('export_hierarchical_extent_candidates', []),
        ('export_native_support_overlap', ['--fallback-only']),
        ('fit_source_scope_gate', ['--hard-pairs']),
        ('export_rank_preserving_completion', ['--within-extent']),
        ('fit_exact_render_query_threshold', []),
    ]
    no_device = {'export_hierarchical_extent_candidates', 'export_native_support_overlap', 'export_rank_preserving_completion'}
    marker = root/'complete_readout_execution_20260917.json'
    record = dict(scene=a.scene, source_evidence_reused=True, field_dependent_heads_rebuilt=True,
        source_evidence_sha256=sha256_file(root/'caption_region_evidence.pt'), commands=[], completed=[])
    snapshots = root/'complete_readout_sources'
    snapshots.mkdir(exist_ok=False)
    for name, extra in stages:
        script = Path('scripts/audits')/(name+'.py')
        shutil.copy2(script, snapshots/script.name)
        command = [sys.executable, str(script), '--scene', a.scene, '--root', str(root), *extra]
        if name not in no_device:
            command += ['--device', a.device]
        record['commands'].append(command)
    record['source_snapshots'] = {str(p): sha256_file(p) for p in snapshots.glob('*.py')}
    marker.write_text(json.dumps(record, indent=2)+'\n')
    for i, command in enumerate(record['commands']):
        print('starting', i, stages[i][0], flush=True)
        with (root/f'complete_readout_stage_{i}.log').open('x') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS='4', PYTHONPATH='.'))
        record['completed'].append(i)
        marker.write_text(json.dumps(record, indent=2)+'\n')
        print('completed', i, flush=True)


if __name__ == '__main__':
    main()
