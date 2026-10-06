"""Dataset adapter: admit a connected source validation pair before training.

All raw VFM tensors remain exact. Split repair uses source pseudo-correspondence
availability only, never field metrics or benchmark labels. It is not an
independent final-test selection procedure.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--geometry', type=Path, required=True)
p.add_argument('--device', required=True)
a, remaining = p.parse_known_args()
root = a.root
provisional = root/'teachers.provisional.json'
manifest = root/'teachers.json'
if not manifest.exists():
    from radio_gs.scripts.prepare_four_teacher_retrieval import main
    sys.argv = ['prepare_four_teacher_retrieval', '--root', str(root), '--device', a.device, *remaining]
    main()
    gc.collect()
    torch.cuda.empty_cache()
if not provisional.exists():
    shutil.copy2(manifest, provisional)
info = json.loads(provisional.read_text())
assert info['authority_sha256'] == sha256_file(root/'responsibility.json')
assert info['source_manifest_sha256'] == sha256_file(root/'source_frames.json')
assets = {r['path']:r['sha256'] for r in info['teacher_files'].values()}
assets[info['query_file']['path']] = info['query_file']['sha256']
for path, digest in assets.items():
    assert sha256_file(root/path) == digest
diagnostic = root/'source_split_admission.json'
if not diagnostic.exists():
    assert sha256_file(manifest) == sha256_file(provisional)
    subprocess.run([sys.executable, 'scripts/audits/run_scannet_source_stage.py',
        'scripts/audits/diagnose_registered_source_split.py', '--root', str(root),
        '--geometry', str(a.geometry), '--device', a.device, '--source-split-diagnostic', str(diagnostic)], check=True)
d = json.loads(diagnostic.read_text())
assert d['source_only'] and not d['benchmark_labels_used']
assert d['teacher_manifest_sha256'] == sha256_file(provisional)
selected = d['selected']['validation_frames']
info['requested_validation_frames'] = info['validation_frames']
info['validation_frames'] = selected
info['train_frames'] = [f for f in info['frames'] if f not in selected]
info['source_split_admission'] = dict(path=str(diagnostic), sha256=sha256_file(diagnostic),
    provisional_manifest_sha256=sha256_file(provisional),
    implementation_sha256=sha256_file(__file__), raw_teacher_tensors_unchanged=True)
manifest.write_text(json.dumps(info, indent=2)+'\n')
for path, digest in assets.items():
    assert sha256_file(root/path) == digest
print('effective source split', info['train_frames'], selected, flush=True)
