"""Resume a verified OOM without changing frozen training math or sample count."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from radio_gs.utils.immutable_artifacts import sha256_file

p = argparse.ArgumentParser()
p.add_argument('--scene', choices=['teatime', 'waldo_kitchen'], required=True)
a = p.parse_args()
root = Path('output/depth_opacity_pipeline_20260918')/a.scene
base = root/'base'
marker = base/'execution.json'
d = json.loads(marker.read_text())
row = next(r for r in d['stages'] if r['name'] == 'field')
assert row['status'] == 'failed' and 'OutOfMemoryError' in (base/'field.log').read_text()
history = base/'failed_attempts/field_oom_before_free_gpu_20260919'
history.mkdir(parents=True, exist_ok=False)
shutil.copy2(marker, history/'execution.json')
shutil.move(str(base/'field.log'), str(history/'field.log'))
shutil.copytree(base/'sources/field', history/'sources')
partial = base/'field/ropefix96_protected'
assert partial.exists() and not list(partial.iterdir()), 'Inspect partial checkpoints before recovery'
shutil.move(str(partial), str(history/'empty_output_directory'))
wrapper = 'scripts/audits/run_source_with_saved_tensor_offload.py'
actual = row['command']
row.update(status='running', actual_command=actual,
    execution_memory_policy='dedicated free GPU; original frozen command, no tensor offload',
    memory_wrapper_sha256=sha256_file(wrapper), failed_attempt=str(history))
marker.write_text(json.dumps(d, indent=2)+'\n')
with (base/'field.log').open('x') as log:
    ret = subprocess.run(actual, stdout=log, stderr=subprocess.STDOUT,
        env=dict(os.environ, OMP_NUM_THREADS='4', PYTHONPATH='.'))
row.update(returncode=ret.returncode, status='failed' if ret.returncode else 'complete')
if not ret.returncode:
    report = base/'field/ropefix96_protected/report.json'
    row['outputs'] = {str(report):sha256_file(report)}
marker.write_text(json.dumps(d, indent=2)+'\n')
if ret.returncode:
    raise RuntimeError('Offloaded source stage failed; inspect preserved new attempt')
outer = root/'execution.json'
record = json.loads(outer.read_text())
failed = next(r for r in record['stages'] if r['name'] == 'base')
assert failed['status'] == 'failed'
hist = root/'failed_attempts/base_field_oom_before_free_gpu_20260919'
hist.mkdir(parents=True, exist_ok=False)
shutil.copy2(outer, hist/'execution.json')
for path in [root/'base.log', root/'sources/base']:
    if path.exists():
        shutil.move(str(path), str(hist/path.name))
record['stages'].remove(failed)
record.setdefault('failed_attempts', []).append(str(hist))
outer.write_text(json.dumps(record, indent=2)+'\n')
commands = [[sys.executable, 'scripts/audits/run_depth_opacity_semantic_rebuild.py', '--scene', a.scene, '--resume']]
field = root/'reconstruction/field'
guard = 'scripts/audits/run_lerf_source_stage.py'
commands += [
    [sys.executable, guard, 'scripts/audits/fit_bounded_fallback_scope.py', '--root', str(field), '--device', 'cuda:0', '--withheld-concepts'],
    [sys.executable, guard, 'scripts/audits/fit_counterfactual_extent_admission.py', '--root', str(field), '--scene', a.scene,
     '--device', 'cuda:0', '--parent', 'bounded_fallback_withheld', '--name', 'bounded_fallback_admission', '--withheld-fallback'],
    [sys.executable, 'scripts/audits/evaluate_bounded_fallback_repair.py', '--scene', a.scene],
]
for i, cmd in enumerate(commands):
    with (history/f'continuation_{i}.log').open('x') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True,
            env=dict(os.environ, OMP_NUM_THREADS='4', PYTHONPATH='.'))
    if i == 1:
        assert json.loads((field/'bounded_fallback_withheld/report.json').read_text())['source_acceptance']
(history/'completed.json').write_text(json.dumps(dict(complete=True, commands=commands,
    final_field_sha256=sha256_file(field/'bounded_fallback_admission/field.pt')), indent=2)+'\n')
