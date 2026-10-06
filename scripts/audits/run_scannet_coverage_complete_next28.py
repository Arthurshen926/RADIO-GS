"""Refit every dependent semantic head after the fixed source-view change."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--attempt-name', default='complete_next28')
    args = parser.parse_args()
    scene = args.scene
    if not args.attempt_name.replace('_', '').isalnum():
        raise ValueError('attempt-name must contain only letters, digits or underscore')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Explicit CUDA_VISIBLE_DEVICES required')
    root = (Path('output/scannet_coverage_native_probe_20260919') if scene in ('scene0000_00', 'scene0062_00')
            else Path('output/scannet_coverage_native_next28')) / scene
    native = json.loads((root/'execution.json').read_text())
    if not native['complete']:
        raise RuntimeError('Native source reconstruction must finish first')
    full_base = root/'base_full_next28'
    base_receipt = json.loads((full_base/'execution.json').read_text())
    if not base_receipt.get('base_finished'):
        raise RuntimeError('Complete source/base reconstruction must finish first')
    required = ('caption_grounded_regions.json', 'caption_region_evidence.pt',
                'caption_region_evidence.json', 'grounded_text_supervision.pt',
                'source_phrase_groups.json')
    missing = [name for name in required if not (full_base/'field'/name).exists()]
    if missing:
        raise RuntimeError('Native-only base cannot enter complete readout rebuild; '
                           'run the complete registered source/base pipeline first. Missing: '
                           + ', '.join(missing))
    out = root/'reconstruction_complete_next28'
    marker = root/(args.attempt_name+'_execution.json')
    if marker.exists():
        raise FileExistsError('Inspect preserved full-rebuild execution before retry: '+str(marker))
    record = dict(scene=scene, native_execution_sha256=sha256_file(root/'execution.json'),
                  complete_base_execution_sha256=sha256_file(full_base/'execution.json'),
                  visibility_report_sha256=sha256_file(Path('output/scannet_visibility_audit_next28')/scene/'report.json'),
                  complete=False, stages=[])

    def save():
        marker.write_text(json.dumps(record, indent=2)+'\n')

    def stage(name, command, outputs):
        command = list(map(str, command))
        row = dict(name=name, command=command, status='running')
        record['stages'].append(row)
        save()
        with (root/(name+'_'+args.attempt_name+'.log')).open('x') as log:
            ret = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='4'))
        row.update(returncode=ret.returncode, status='complete' if ret.returncode == 0 else 'failed')
        if ret.returncode == 0:
            row['outputs'] = {str(path): sha256_file(path) for path in outputs}
        save()
        if ret.returncode:
            raise RuntimeError(f'{scene} {name} failed; inspect preserved log')

    base = full_base/'field'
    gp = root/'geometry_report.json'
    field = out/'field'
    stage('rebuild', [sys.executable, 'scripts/audits/run_registered_mesh_pixel_rebuild.py',
                      '--scene', scene, '--device', args.device, '--parent', base,
                      '--output-root', out, '--geometry-report', gp, '--skip-evaluation'],
          [field/'joint_supervised_image_residual/report.json'])
    guard = 'scripts/audits/run_scannet_source_stage.py'
    stage('bounded_scope', [sys.executable, guard, 'scripts/audits/fit_bounded_fallback_scope.py',
                            '--root', field, '--device', args.device, '--withheld-concepts'],
          [field/'bounded_fallback_withheld/report.json'])
    record['source_acceptance'] = json.loads((field/'bounded_fallback_withheld/report.json').read_text())['source_acceptance']
    save()
    stage('admission', [sys.executable, guard, 'scripts/audits/fit_counterfactual_extent_admission.py',
                        '--root', field, '--scene', scene, '--device', args.device,
                        '--parent', 'bounded_fallback_withheld', '--name', 'bounded_fallback_admission',
                        '--withheld-fallback'],
          [field/'bounded_fallback_admission/report.json'])
    record['complete'] = True
    record['independent_test'] = False
    record['full_method_rebuilt'] = True
    save()


if __name__ == '__main__':
    main()
