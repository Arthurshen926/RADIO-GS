"""Rebuild the whole semantic pipeline for source-selected larger view budgets."""
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
    a = p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Explicit GPU binding required')
    root = Path('output/scannet_expanded_budget_next29') / a.scene
    base = root / 'base'
    plan = json.loads((base / 'source_plan.json').read_text())
    marker = root / 'execution.json'
    record = json.loads(marker.read_text())
    if record['stages']:
        raise RuntimeError('Preserve and inspect earlier attempt before retry')
    record.update(implementation_sha256=sha256_file(__file__),
                  source_only_reconstruction=True, larger_source_budget=True,
                  source_plan_sha256=sha256_file(base / 'source_plan.json'))

    def save():
        temp = marker.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2)+'\n')
        temp.replace(marker)

    def stage(name, command):
        row = dict(name=name, command=list(map(str, command)), status='running')
        record['stages'].append(row)
        save()
        try:
            with (root / (name+'.log')).open('x') as log:
                result = subprocess.run(row['command'], stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='4'))
            row['returncode'] = result.returncode
            if result.returncode:
                raise RuntimeError(f'{a.scene} {name} failed')
            row['status'] = 'complete'
        except Exception as error:
            row.update(status='failed', error=repr(error))
            save()
            raise
        save()
        print(a.scene, name, 'complete', flush=True)

    py = sys.executable
    guard = 'scripts/audits/run_scannet_source_stage.py'
    stage('radio', [py, guard, '--module', 'radio_gs.scripts.extract_radio_features',
        '--scene', a.scene, '--image_dir', root/'source_rgb', '--output_dir', root/'radio',
        '--radio_checkpoint', '/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar',
        '--batch_size', '1', '--skip_pca_stats', '--resume-partial', '--device', 'cuda:0'])
    command = ['/root/miniconda3/envs/sam3/bin/python', guard, '--module',
        'radio_gs.scripts.build_sam3_multiscale_hierarchy_cache',
        '--source-authority', plan['source_authority']['path'], '--source-authority-sha256',
        plan['source_authority']['sha256'], '--image-ids',
        ','.join(f'frame_{f:05d}' for f in plan['frames']), '--output-root', base/'source_sam',
        '--checkpoint-path', 'checkpoints/sam3_modelscope/sam3.pt', '--checkpoint-sha256',
        '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e', '--device', 'cuda:0']
    stage('sam', command)
    snapshot = base/'sam_sources'
    snapshot.mkdir()
    bindings = {}
    for name in ['radio_gs/scripts/build_sam3_multiscale_hierarchy_cache.py',
                 'radio_gs/scripts/build_sam3_foundation_cache.py',
                 'radio_gs/models/sam3_multiscale_hierarchy.py', guard]:
        shutil.copy2(name, snapshot/Path(name).name)
        bindings[name] = sha256_file(name)
    (base/'source_sam_execution.json').write_text(json.dumps(dict(status='complete',
        command=list(map(str, command)), implementations=bindings), indent=2)+'\n')
    stage('complete_base', [py, '-m', 'scripts.audits.run_registered_mesh_base',
        '--scene', a.scene, '--device', 'cuda:0', '--output-root', base,
        '--geometry-report', root/'geometry_report.json', '--until-base'])
    reconstruction = root/'reconstruction'
    stage('complete_readout', [py, '-m', 'scripts.audits.run_registered_mesh_pixel_rebuild',
        '--scene', a.scene, '--device', 'cuda:0', '--parent', base/'field',
        '--output-root', reconstruction, '--geometry-report', root/'geometry_report.json',
        '--skip-evaluation'])
    field = reconstruction/'field'
    stage('bounded_scope', [py, guard, 'scripts/audits/fit_bounded_fallback_scope.py',
        '--root', field, '--device', 'cuda:0', '--withheld-concepts'])
    scope = json.loads((field/'bounded_fallback_withheld/report.json').read_text())
    record['source_acceptance'] = scope['source_acceptance']
    # Preserve the final diagnostic model, but inference must enforce both receipts.
    stage('admission', [py, guard, 'scripts/audits/fit_counterfactual_extent_admission.py',
        '--root', field, '--scene', a.scene, '--device', 'cuda:0',
        '--parent', 'bounded_fallback_withheld', '--name', 'bounded_fallback_admission',
        '--withheld-fallback'])
    from radio_gs.querying.source_accepted_category import category_scope_decision
    admission = json.loads((field/'bounded_fallback_admission/report.json').read_text())
    record.update(complete=True, full_method_rebuilt=True,
                  deployment_decision=category_scope_decision(scope, admission),
                  final_field_sha256=admission['field_sha256'],
                  runtime_sha256=sha256_file('radio_gs/querying/source_accepted_category.py'))
    save()


if __name__ == '__main__':
    main()
