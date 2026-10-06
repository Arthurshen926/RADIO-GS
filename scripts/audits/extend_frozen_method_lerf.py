"""Apply the frozen two-scene recipe to the remaining LERF scenes.

Only dataset preparation is new. Preserve source splits and raw 2D SAM assets;
rebuild every geometry-indexed feature, correspondence, head and readout.
"""
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
    p.add_argument('--scene', choices=['teatime', 'waldo_kitchen'], required=True)
    a = p.parse_args()
    root = Path('output/frozen_method_extension_20260918')/a.scene
    root.mkdir(parents=True, exist_ok=False)
    freeze = json.loads(Path('paper/artifacts/frozen_cross_benchmark_method_20260918.json').read_text())
    source_manifest = json.loads(Path(freeze['source_manifest']).read_text())
    for path, digest in source_manifest.items():
        if sha256_file(path) != digest:
            raise ValueError('Frozen method implementation changed: '+path)
    record = dict(method_id=freeze['method_id'], scene=a.scene, complete=False, stages=[])
    def run(entry, args, source=False):
        cmd = [sys.executable]+(['scripts/audits/run_lerf_source_stage.py'] if source else [])+[entry]+list(map(str, args))
        index = len(record['stages'])
        row = dict(command=cmd, status='running')
        record['stages'].append(row)
        (root/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
        with (root/f'{index:02d}.log').open('x') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='4'))
        row.update(returncode=result.returncode, status='failed' if result.returncode else 'complete')
        (root/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'Stage {index} failed; inspect preserved log')
    run('scripts/audits/fit_source_depth_opacity.py', ['--scene', a.scene], True)
    oldpath = Path('output/geometry_only_repair_final_20260914')/a.scene/'report.json'
    geo = Path('output/depth_opacity_repair_20260918')/a.scene
    fitted = json.loads((geo/'report.json').read_text())
    old = json.loads(oldpath.read_text())
    assert fitted['parent_report_sha256'] == sha256_file(oldpath)
    spec = old['results']['independent_scale']
    old['results']['independent_scale'] = dict(fitted['after'], geometry=fitted['geometry'],
        geometry_sha256=fitted['geometry_sha256'], elements=spec['elements'])
    old['opacity_preprocessing'] = dict(report=str((geo/'report.json').resolve()),
        sha256=sha256_file(geo/'report.json'), parent_report_sha256=sha256_file(oldpath), independent_depth_only=True)
    (geo/'geometry_report.json').write_text(json.dumps(old, indent=2)+'\n')
    source = Path('output/lerf_complete_v2_20260918')/a.scene/'source'
    base = Path('output/depth_opacity_pipeline_20260918')/a.scene/'base'
    base.mkdir(parents=True, exist_ok=False)
    for name in ('source_plan.json', 'source_sam_execution.json'):
        shutil.copy2(source/name, base/name)
    for name in ('source_sam', 'sam_sources'):
        (base/name).symlink_to((source/name).resolve(), target_is_directory=True)
    run('scripts/audits/run_depth_opacity_semantic_rebuild.py', ['--scene', a.scene])
    field = base.parent/'reconstruction/field'
    run('scripts/audits/fit_bounded_fallback_scope.py', ['--root', field, '--device', 'cuda:0', '--withheld-concepts'], True)
    fit = json.loads((field/'bounded_fallback_withheld/report.json').read_text())
    if not fit['source_acceptance']:
        raise RuntimeError('Frozen fallback admission failed on source validation; no target-driven override')
    run('scripts/audits/fit_counterfactual_extent_admission.py', ['--root', field, '--scene', a.scene,
        '--device', 'cuda:0', '--parent', 'bounded_fallback_withheld', '--name', 'bounded_fallback_admission', '--withheld-fallback'], True)
    run('scripts/audits/evaluate_bounded_fallback_repair.py', ['--scene', a.scene])
    record['complete'] = True
    (root/'execution.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
