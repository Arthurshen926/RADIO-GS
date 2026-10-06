"""Reconstruct all geometry-dependent semantic assets for the opacity control."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', required=True); p.add_argument('--resume', action='store_true'); args = p.parse_args()
    root = Path('output/depth_opacity_pipeline_20260918')/args.scene
    gp = Path('output/depth_opacity_repair_20260918')/args.scene/'geometry_report.json'
    base = root/'base'; finalroot = root/'reconstruction'; field = finalroot/'field'
    marker = root/'execution.json'
    if marker.exists():
        assert args.resume; record = json.loads(marker.read_text()); assert record['geometry_report_sha256'] == sha256_file(gp)
    else:
        record = dict(schema='depth_opacity_semantic_rebuild.v1', scene=args.scene, geometry_report_sha256=sha256_file(gp), reused_2d_source_sam=True, reused_geometry_indexed_assets=False, stages=[])
    def run(name, entry, extra, outputs, source=False):
        command = [sys.executable]+(['scripts/audits/run_lerf_source_stage.py'] if source else [])+[entry]+list(map(str, extra))
        old = next((r for r in record['stages'] if r['name'] == name), None)
        if old:
            assert old['status'] == 'complete' and old['command'] == command, 'Inspect failed stage before resume'
            for path, digest in old['outputs'].items(): assert sha256_file(path) == digest
            return
        snap = root/'sources'/name; snap.mkdir(parents=True)
        sources = [Path(__file__), Path(entry)]
        impl = {}
        for path in sources: shutil.copy2(path, snap/path.name); impl[str(path)] = sha256_file(path)
        actual_command = command + (['--resume'] if name == 'pixel_reconstruction' and (finalroot/'execution.json').exists() else [])
        row = dict(name=name, command=command, actual_command=actual_command, implementations=impl, status='running'); record['stages'].append(row); marker.write_text(json.dumps(record, indent=2)+'\n')
        print(args.scene, name, flush=True)
        with (root/(name+'.log')).open('x') as log:
            r = subprocess.run(actual_command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='4'))
        row.update(returncode=r.returncode, status='failed' if r.returncode else 'complete')
        if r.returncode == 0: row['outputs'] = {str(path):sha256_file(path) for path in outputs}
        marker.write_text(json.dumps(record, indent=2)+'\n')
        if r.returncode: raise RuntimeError(name+' failed; inspect '+str(root/(name+'.log')))
    run('base', 'scripts/audits/run_pose_covered_lerf_pipeline.py', ['--scene', args.scene, '--device', 'cuda:0', '--output-root', base, '--geometry-report', gp, '--until-base'], [base/'execution.json'])
    run('pixel_reconstruction', 'scripts/audits/run_pixel_evidence_rebuild.py', ['--scene', args.scene, '--device', 'cuda:0', '--parent', base/'field', '--output-root', finalroot, '--geometry-report', gp, '--skip-evaluation'], [finalroot/'execution.json'])
    final = 'counterfactual_extent_benefit_admission'
    run('admission', 'scripts/audits/fit_counterfactual_extent_admission.py', ['--root', field, '--scene', args.scene, '--device', 'cuda:0'], [field/final/'report.json'], source=True)
    for task in ['text', 'reference']:
        dest = root/(task+'_online.json')
        common = ['--task', task, '--root', field, '--scene', args.scene, '--geometry-report', gp, '--runs', final, '--device', 'cuda:0']
        run('evaluate_'+task, 'scripts/audits/audit_current_query_routes.py', [*common, '--output', dest], [dest, dest.with_suffix('.encodings.pt')])
        replay = root/(task+'_replay.json')
        run('replay_'+task, 'scripts/audits/audit_current_query_routes.py', [*common, '--encoding-replay', dest.with_suffix('.encodings.pt'), '--output', replay], [replay])
    record['complete'] = True; marker.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
