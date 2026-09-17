"""Finish the nine-stage range readout with replacement-native evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    root = base/'openclip_replacement_20260917'/a.scene
    marker = root/'functional_complete_execution.json'
    if marker.exists():
        raise FileExistsError(marker)
    stages = [
        ('build_functional_caption_evidence', []),
        ('train_bounded_structure_exploration', ['--root', str(root), '--geometry', str(base/'geometry_only_repair_final_20260914'/a.scene/'independent_scale.pt'), '--parent', 'openclip96_protected_imagecal_balanced', '--variant', 'explicit', '--passes', '0', '--name', 'functional_affinity']),
        ('fit_full_source_query_threshold', []),
        ('fit_grounded_semantic_transfer', ['--group-report', 'functional_membership_groups.json']),
        ('train_source_region_extents', []),
        ('train_rendered_extent', ['--residual-dim', '128', '--steps', '800']),
        ('export_hierarchical_extent_candidates', []),
        ('export_native_support_overlap', ['--fallback-only']),
        ('fit_source_scope_gate', ['--hard-pairs']),
        ('export_rank_preserving_completion', ['--within-extent']),
        ('fit_exact_render_query_threshold', []),
    ]
    snapshots = root/'functional_complete_sources'
    snapshots.mkdir()
    record = dict(scene=a.scene, source_only=True, benchmark_data_used=False, stages=[])
    for name, extra in stages:
        script = Path('scripts/audits')/(name+'.py')
        shutil.copy2(script, snapshots/script.name)
        command = [sys.executable, str(script)]
        if name != 'train_bounded_structure_exploration':
            command += ['--scene', a.scene]
        if name not in ('build_functional_caption_evidence', 'train_bounded_structure_exploration'):
            command += ['--root', str(root)]
        if not name.startswith('export_'):
            command += ['--device', a.device]
        command += extra
        row = dict(name=name, command=command, implementation_sha256=sha256_file(script), status='running')
        record['stages'].append(row)
        marker.write_text(json.dumps(record, indent=2)+'\n')
        print(a.scene, name, flush=True)
        with (root/(name+'_functional.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS='4', PYTHONPATH='.'))
        row['returncode'] = result.returncode
        row['status'] = 'failed' if result.returncode else 'complete'
        marker.write_text(json.dumps(record, indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(name+' failed')
        if name == 'train_bounded_structure_exploration':
            shutil.copytree(root/'functional_affinity_direct_control', root/'scale_affinity_v1_direct_control')
    record['finished'] = True
    marker.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
