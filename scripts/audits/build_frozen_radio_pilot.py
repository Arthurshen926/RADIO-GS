"""Reproducible four-view production teacher-builder smoke run."""
import json, subprocess, sys, hashlib
from pathlib import Path
root = Path('/mnt/pool/sqy/results/RADIO-GS/output')
r = root/'geometry_only_repair_final_20260914/figurines/report.json'
g = json.loads(r.read_text())['results']['independent_scale']
m = json.loads((root/'canonical_teacher_features/figurines/frame_manifest.json').read_text())
out = root/'frozen_radio_teacher_pilot_20260915/figurines'
options = {
'config':'radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml',
'checkpoint':g['geometry'], 'frozen-geometry':g['geometry'],
'expected-frozen-geometry-sha256':g['geometry_sha256'],
'expected-geometry-checkpoint-sha256':g['geometry_sha256'],
'frozen-geometry-report':str(r),
'expected-frozen-geometry-report-sha256':hashlib.sha256(r.read_bytes()).hexdigest(),
'expected-feature-output-bundle-sha256':m['output_bundle_sha256'],
'output':str(out/'radio.pt'), 'save-responsibility-cache':str(out/'responsibility.json'),
'observation-contract':'legacy', 'aggregation-mode':'raster_marginal_responsibility',
'raster-view-fusion':'contribution_mean', 'feature-space':'radio', 'max-views':'4',
'include-frame-ids':'3,6,8,11', 'raster-channel-chunk-size':'128',
'raster-reliability-mode':'mean_resultant', 'alpha-threshold':'0'}
cmd = [sys.executable, '-m', 'radio_gs.scripts.build_gaussian_multiview_teacher_cache']
for k,v in options.items():
    cmd.extend(['--'+k,v])
cmd.append('--normalize-each-view')
subprocess.run(cmd, check=True)
