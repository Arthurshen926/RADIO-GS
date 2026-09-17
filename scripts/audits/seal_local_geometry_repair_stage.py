"""Seal local geometric correction and downstream native-only controls."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry

out=Path('paper/artifacts/local_geometry_repair_stage_seal_20260916.json')
if out.exists():raise FileExistsError(out)
base=Path('/mnt/pool/sqy/results/RADIO-GS/output');artifacts=[];sources={};checks={}
def add(p):
 p=Path(p);artifacts.append(dict(path=str(p),sha256=sha256_file(p)))
def source(path):sources[path]=sha256_file(path)
for scene in ['figurines','ramen']:
 parent=base/'preserved_source_inventory_20260916'/scene/'exact_render_query_threshold96';pr=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==pr['field_sha256']
 checks[scene]=dict(complete_mainline_unchanged=True,field_sha256=pr['field_sha256'],complete_method_promoted=False)
 for folder,script in [('surface_consensus_controls_20260916','refine_surface_depth_consensus.py'),('trusted_source_geometry_20260916','rebuild_trusted_source_geometry.py'),('local_calibrated_geometry_20260916','rebuild_local_calibrated_geometry.py')]:
  root=base/folder/scene;p=root/'report.json';r=json.loads(p.read_text());code='scripts/audits/'+script
  assert r['implementation_sha256']==sha256_file(code) and not r['benchmark_data_used'] and not r['promoted'];source(code);add(p)
  model=FrozenGaussianGeometry.load(r['candidate_geometry'],expected_sha256=r['candidate_geometry_sha256']);add(r['candidate_geometry'])
  if 'soft_geometry' in r:FrozenGaussianGeometry.load(r['soft_geometry'],expected_sha256=r['soft_geometry_sha256']);add(r['soft_geometry'])
  if folder=='local_calibrated_geometry_20260916':
   assert r['local_implementation_sha256']==sha256_file('radio_gs/geometry/local_depth_calibration.py')
   for f in ['local_calibration.pt','authority_geometry_report.json','native_rebuilt_diagnostic.pt','native_diagnostic_authority/responsibility.json','native_diagnostic_authority/build.json']:add(root/f)
   auth=json.loads((root/'native_diagnostic_authority/responsibility.json').read_text());model.assert_binding(auth['metadata']['frozen_geometry_binding'])
   for rec in auth['views']:
    pp=root/'native_diagnostic_authority'/rec['relative_path'];assert sha256_file(pp)==rec['sha256']
   rebuilt=torch.load(root/'native_rebuilt_diagnostic.pt',weights_only=False);model.assert_binding(rebuilt['geometry_binding'])
 for suffix,script in [('surface_support_vs_render','audit_surface_support_vs_render.py'),('source_local_scale_final','audit_source_local_scale.py'),('local_geometry_native_rebuild','evaluate_local_geometry_native_rebuild.py')]:
  p=Path('paper/artifacts')/f'{scene}_{suffix}_20260916.json';r=json.loads(p.read_text());code='scripts/audits/'+script;assert r['implementation_sha256']==sha256_file(code);source(code);add(p)
  if 'local_implementation_sha256' in r:assert r['local_implementation_sha256']==sha256_file('radio_gs/geometry/local_depth_calibration.py')
  if suffix=='local_geometry_native_rebuild':assert not r['complete_method'] and not r['target_rgb_used']
for p in ['radio_gs/geometry/local_depth_calibration.py','radio_gs/geometry/source_scale_calibration.py','tests/test_local_depth_calibration.py','scripts/audits/build_local_geometry_diagnostic_authority.py',__file__]:source(p)
out.write_text(json.dumps(dict(schema='local_geometry_repair_stage_seal.v1',artifacts=artifacts,sources=sources,checks=checks,related_tests_passed=31),indent=2)+'\n');print('sealed',len(artifacts),'artifacts;',len(sources),'sources; geometry cold-load and authority bindings pass')
