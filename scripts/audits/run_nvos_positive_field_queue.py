"""Freeze a common existing image branch, then score all eight positive tasks."""
import json
import os
from pathlib import Path
import subprocess
import sys
from radio_gs.utils.immutable_artifacts import sha256_file

tasks = ['fern', 'flower', 'fortress', 'horns_center', 'horns_left', 'leaves', 'orchids', 'trex']
run = 'image_supervised_native_residual_measured_support512'
root = Path('output/nvos_positive_field_20260918')
root.mkdir(exist_ok=False)
base = Path('output/nvos_ropefix_pipeline_20260917')
bindings = {}
for task in tasks:
    scene = 'horns_center' if task == 'horns_left' else task
    parent = base/scene
    assert json.loads((parent/'execution.json').read_text())['finished']
    for path in [parent/'field'/run/'field.pt', parent/'field'/run/'report.json', parent/'frozen_geometry/report.json']:
        bindings[str(path)] = sha256_file(path)
record = dict(contract='nvos-ludvig-online-all-view-eight-v1', tasks=tasks, run=run,
    independent_test=False, full_lerf_caption_method=False,
    branch='Previously sealed native96 NVOS image branch; no per-task variant selection',
    allowed_but_unused_inputs='target RGB and negative-independent all-view uplifting',
    negative_prompt_allowed=False, bindings=bindings, stages=[], complete=False)
(root/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
for task in tasks:
    parent = base/('horns_center' if task == 'horns_left' else task)
    cmd = [sys.executable, 'scripts/audits/evaluate_nvos_positive_field.py', '--scene', task,
        '--root', str(parent/'field'), '--geometry-report', str(parent/'frozen_geometry/report.json'),
        '--runs', run, '--query-root', 'output/nvos_positive_queries_20260918', '--device', 'cuda:0',
        '--output', str(root/task)]
    row = dict(task=task, command=cmd, status='running')
    record['stages'].append(row)
    (root/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
    with (root/(task+'.log')).open('x') as log:
        ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS='4', PYTHONPATH='.'))
    row.update(returncode=ret.returncode, status='failed' if ret.returncode else 'complete')
    (root/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
    if ret.returncode:
        raise RuntimeError(task+' failed; preserve this attempt')
results = {}
for task in tasks:
    p = root/task/'report.json'
    r = json.loads(p.read_text())
    assert r['negative_scribble_opened'] is False and not r['signed_refinement']
    results[task] = r['results'][run]
for path, digest in bindings.items():
    assert sha256_file(path) == digest
record.update(complete=True, results=results,
    macro={k:sum(r[k] for r in results.values())/len(tasks) for k in ['iou', 'pixel_accuracy', 'average_precision']})
(root/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
print(record['macro'], flush=True)
