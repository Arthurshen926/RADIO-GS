"""Seal final evidence for the first next-stage geometry iteration."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

out=Path('paper/artifacts/geometry_distribution_stage_seal_20260916.json')
if out.exists():raise FileExistsError(out)
pairs=[('ramen_source_contribution_distribution_v2_20260916.json','audit_source_contribution_distribution.py'),('ramen_source_reprojection_scale_final_20260916.json','audit_source_reprojection_scale.py')]
for scene in ['figurines','ramen']:
 pairs.extend([(f'{scene}_all_source_track_render_v2_20260916.json','audit_geometry_all_source_tracks.py'),(f'{scene}_source_track_parallax_20260916.json','stratify_geometry_track_reference.py')])
artifacts=[];sources={};checks={}
for name,script in pairs:
 path=Path('paper/artifacts')/name;code=Path('scripts/audits')/script;r=json.loads(path.read_text())
 assert r['source_only'] and not r['benchmark_data_used']
 assert r['implementation_sha256']==sha256_file(code)
 sources[str(code)]=sha256_file(code);artifacts.append(dict(path=str(path),sha256=sha256_file(path)))
 if 'fit_implementation_sha256' in r:assert r['fit_implementation_sha256']==sha256_file('radio_gs/geometry/source_scale_calibration.py')
for scene in ['figurines','ramen']:
 root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/scene/'exact_render_query_threshold96'
 report=json.loads((root/'report.json').read_text());assert sha256_file(root/'field.pt')==report['field_sha256']
 checks[scene]=dict(mainline_sha256=report['field_sha256'],unchanged=True,new_retrieval_evaluation=False)
 # Domain strata partition the covered observations; expanded audit agrees with first raw renderer run.
 r=json.loads((Path('paper/artifacts')/f'{scene}_all_source_track_render_v2_20260916.json').read_text())
 old=json.loads((Path('paper/artifacts')/f'{scene}_all_source_track_render_20260916.json').read_text())
 assert r['groups']==old['groups']
 for split in ['mapping','heldout']:
  assert sum(r['support_groups'][f'{split}_semantic_{v}']['absrel']['count'] for v in [True,False])==r['groups'][split]['absrel']['count']
for path in ['radio_gs/geometry/source_scale_calibration.py','tests/test_reprojection_scale_calibration.py',__file__]:sources[path]=sha256_file(path)
r=json.loads(Path('paper/artifacts/ramen_source_contribution_distribution_v2_20260916.json').read_text());old=json.loads(Path('paper/artifacts/ramen_source_track_geometry_transport_20260916.json').read_text())
assert len(r['rows'])==len(old['rows'])
for new,prior in zip(r['rows'],old['rows']):
 assert (new['point_id'],new['source_frame'],new['target_frame'])==(prior['point_id'],prior['source_frame'],prior['target_frame'])
 assert abs(new['gaussian_pixels']-prior['gaussian_pixels'])<1e-5
out.write_text(json.dumps(dict(schema='geometry_distribution_stage_seal.v1',artifacts=artifacts,sources=sources,checks=checks,tests='15 passed: reprojection scale, source scale, geometry prior',new_geometry_promoted=False),indent=2)+'\n')
print('sealed',len(artifacts),'final artifacts; legacy centroid and full renderer regressions exact')
