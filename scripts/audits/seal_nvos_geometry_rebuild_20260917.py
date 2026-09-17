"""Seal real rebuilt-field evaluations and source-only scale-bound controls."""
import json,shutil
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
base=Path('/mnt/pool/sqy/results/RADIO-GS/output');out=Path('paper/artifacts/nvos_geometry_rebuild_seal_20260917.json');assert not out.exists();artifacts={};sources={};evaluations={}
def bind(p):artifacts[str(p)]=sha256_file(p);return json.loads(Path(p).read_text())
for scene,dirname in [('fortress','nvos_scale_pipeline_20260917'),('fern','nvos_local_scale_pipeline_20260917')]:
 root=base/dirname/scene;execution=bind(root/'execution.json');assert execution['finished']
 for name,stage in execution['stages'].items():
  assert stage['status']=='complete'
  for f,digest in stage['outputs'].items():assert sha256_file(f)==digest;artifacts[f]=digest
  for f,digest in stage['implementations'].items():assert sha256_file(root/'sources'/name/Path(f).name)==digest
 for name in ['nvos96_protected','nvos96_protected_imagecal_balanced','image_supervised_native_residual','nvos96_protected_imagecal_balanced_measured_support512','image_supervised_native_residual_measured_support512']:
  r=bind(root/'field'/name/'report.json');p=root/'field'/name/'field.pt';assert sha256_file(p)==r['field_sha256'];artifacts[str(p)]=r['field_sha256']
 evaluations[scene]={}
 for mode in ['image','signed']:
  p=root/'evaluation'/f'{scene}_{mode}'/'report.json';d=bind(p);assert not d['target_rgb_opened'] and d['target_gt_used_after_prediction_only'] and d['training_feature_reads_forbidden'];evaluations[scene][mode]=d['results']
  for run,r in d['results'].items():assert sha256_file(p.parent/(run+'.pt'))==r['prediction_sha256']
for scene in ['flower','orchids']:
 p=base/'nvos_local_scale2_repair_20260917'/scene/'source_scale_prior.json';d=bind(p)
 for c in d['calibration']:
  r=c['source_geometry_override'];assert sha256_file(r['path'])==r['sha256'];artifacts[r['path']]=r['sha256']
 p=base/'nvos_local_scale2_pipeline_20260917'/scene/'frozen_geometry/report.json';d=bind(p)
 for r in d['results'].values():
  if 'geometry' in r:assert sha256_file(r['geometry'])==r['geometry_sha256'];artifacts[r['geometry']]=r['geometry_sha256']
for f in ['nvos_fortress_scale_rebuild_cold','nvos_fern_local_rebuild_cold','nvos_fortress_scale_support_retention','nvos_fern_local_support_retention','nvos_fortress_native_batch_numerics','nvos_fern_native_batch_numerics','nvos_fern_geometry_rebuild_comparison','nvos_flower_local_scale_saturation','nvos_orchids_local_scale_saturation','nvos_leaves_local_scale_saturation','nvos_flower_local_scale2_frozen_distribution','nvos_orchids_local_scale2_frozen_distribution']:
 bind(Path('paper/artifacts')/(f+'_20260917.json'))
files=['radio_gs/scripts/compile_three_query_episodes.py','radio_gs/training/reliable_correspondence.py','radio_gs/training/pipeline_contracts.py','radio_gs/scripts/train_three_query_field.py','scripts/audits/run_nvos_bound_pipeline.py','scripts/audits/run_nvos_source_stage.py','scripts/audits/build_query_free_residual_evidence.py','scripts/audits/audit_local_scale_saturation.py','scripts/audits/build_local_source_scale_prior.py','scripts/audits/compare_nvos_geometry_rebuild.py','scripts/audits/audit_native_batch_numerics.py','scripts/audits/seal_nvos_geometry_rebuild_20260917.py','scripts/audits/evaluate_nvos_frozen_image_route.py','scripts/audits/cold_three_query_native.py','tests/test_pipeline_contracts.py','tests/test_reliable_correspondence.py']
snapshot=base/'nvos_geometry_rebuild_sources_20260917'
for f in files:
 dest=snapshot/f;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.copy2(f,dest);sources[f]=dict(sha256=sha256_file(f),snapshot=str(dest))
out.write_text(json.dumps(dict(schema='nvos_geometry_rebuild_seal.v1',artifacts=artifacts,sources=sources,evaluations=evaluations,claims=dict(complete_fortress_fern_rebuild=True,all_matched_variants_reported=True,independent_final_test=False,complete_joint_benchmark=False,exact_bitwise_cold_equivalence=False,all_signed_constraints_satisfied=False),verdict='Real fern retrieval gains accompany corrected geometry and a full source rebuild. Fortress is newly executable. Global promotion, text extent and the remaining benchmarks are not complete.'),indent=2)+'\n');print(out)
