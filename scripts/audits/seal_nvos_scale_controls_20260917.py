"""Seal executed source geometry controls, including explicit failed scenes."""
import json,shutil
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
base=Path('/mnt/pool/sqy/results/RADIO-GS/output');out=Path('paper/artifacts/nvos_scale_controls_seal_20260917.json');assert not out.exists()
snapshot=base/'nvos_scale_controls_sources_20260917';sources={};artifacts={}
files=['scripts/audits/build_independent_source_scale_prior.py','scripts/audits/build_local_source_scale_prior.py','scripts/audits/audit_nvos_scale_calibration.py','scripts/audits/audit_local_source_scale.py','scripts/audits/audit_frozen_source_scale_geometry.py','scripts/audits/visualize_source_scale_geometry.py','scripts/audits/summarize_nvos_scale_controls.py','scripts/audits/seal_nvos_scale_controls_20260917.py','radio_gs/geometry/independent_view_scale.py','radio_gs/geometry/local_depth_calibration.py','radio_gs/geometry/source_geometry_override.py','radio_gs/field/moge_geometry_prior.py','radio_gs/scripts/repair_moge_geometry.py','tests/test_source_geometry_override.py','tests/test_independent_view_scale.py','tests/test_local_depth_calibration.py','tests/test_affine_depth_normals.py']
for f in files:
 dest=snapshot/f;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.copy2(f,dest);sources[f]=dict(sha256=sha256_file(f),snapshot=str(dest))
for p in Path('paper/artifacts').glob('nvos_*20260917.json'):
 if any(k in p.name for k in ['scale_holdout','frozen_distribution','frozen_geometry_distribution','source_geometry','scale_control_summary']):artifacts[str(p)]=sha256_file(p)
for p in Path('paper/artifacts').glob('nvos_*source_geometry_20260917.png'):artifacts[str(p)]=sha256_file(p)
for scene in ['fern','flower','fortress','horns_center','leaves','orchids','trex']:
 for kind in ['nvos_scale_repair_20260917','nvos_local_scale_repair_20260917']:
  p=base/kind/scene/'source_scale_prior.json'
  if not p.exists():continue
  artifacts[str(p)]=sha256_file(p);d=json.loads(p.read_text())
  for c in d['calibration']:
   if 'source_geometry_override' in c:
    r=c['source_geometry_override'];assert sha256_file(r['path'])==r['sha256'];artifacts[r['path']]=r['sha256']
 for kind in ['nvos_scale_pipeline_20260917','nvos_local_scale_pipeline_20260917']:
  p=base/kind/scene/'frozen_geometry/report.json'
  if not p.exists():continue
  artifacts[str(p)]=sha256_file(p);d=json.loads(p.read_text());assert sha256_file(d['source_report'])==d['source_report_sha256']
  for r in d['results'].values():
   if 'geometry' in r:assert sha256_file(r['geometry'])==r['geometry_sha256'];artifacts[r['geometry']]=r['geometry_sha256']
out.write_text(json.dumps(dict(schema='nvos_scale_control_seal.v1',source_only_geometry_controls=True,sources=sources,artifacts=artifacts,scope='Geometry controls only; full field builds and retrieval have independent stage receipts. Earlier receipts retain their executed code SHA; current source snapshots do not relabel earlier executions.',conclusion='Reject universal global scale cap; accept per-scene candidates only with coverage and tail evidence. Sparse improvement is not dense or retrieval promotion.'),indent=2)+'\n');print(out)
