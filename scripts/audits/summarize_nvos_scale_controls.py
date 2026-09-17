"""Summarize completed geometry controls without dropping failed NVOS scenes."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

base=Path('/mnt/pool/sqy/results/RADIO-GS/output');out=Path('paper/artifacts/nvos_scale_control_summary_20260917.json')
assert not out.exists();rows=[];artifacts={}
def read(path):
 artifacts[str(path)]=sha256_file(path);return json.loads(path.read_text())
for scene in ['fern','flower','fortress','horns_center','leaves','orchids','trex']:
 p=base/'nvos_scale_repair_20260917'/scene/'source_scale_prior.json';pure=read(p)
 row=dict(scene=scene,task_names=['horns_center','horns_left'] if scene=='horns_center' else [scene],pure_scale_accepted=pure['accepted_view_count'],total_views=len(pure['calibration']))
 p=base/'nvos_local_scale_repair_20260917'/scene/'source_scale_prior.json'
 if p.exists():
  local=read(p);row['local_scale_accepted']=local['accepted_view_count']
  for c in local['calibration']:
   if 'source_geometry_override' in c:
    r=c['source_geometry_override'];assert sha256_file(r['path'])==r['sha256']
  row['materialized_maps_hash_verified']=True
  audit=Path(f'paper/artifacts/nvos_{"horns" if scene=="horns_center" else scene}_local_frozen_distribution_20260917.json')
  if audit.exists():row['frozen_geometry_audit']=read(audit)['groups']
  row['bounded_conclusion']='fewer than four admitted views; no complete field build' if local['accepted_view_count']<4 else 'geometry candidate measured; coverage and retrieval acceptance remain separate'
 else:row['bounded_conclusion']='pure-scale candidate; local-scale control unnecessary for this diagnostic'
 rows.append(row)
old=read(Path('paper/artifacts/nvos_fern_old_frozen_distribution_20260917.json'));new=read(Path('paper/artifacts/nvos_fern_local_frozen_distribution_20260917.json'))
left={(r['frame'],r['point_id']):r for r in old['rows'] if r['split']=='fusion_heldout'};right={(r['frame'],r['point_id']):r for r in new['rows'] if r['split']=='fusion_heldout'}
assert left.keys()==right.keys();keys=[k for k in left if left[k]['covered'] and right[k]['covered']];paired={}
for name,data in [('old',left),('local',right)]:
 e=torch.tensor([data[k]['absrel'] for k in keys]);paired[name]=dict(count=len(e),median=float(e.median()),p90=float(e.quantile(.9)))
result=dict(schema='nvos_scale_controls_summary.v1',source_only=True,benchmark_queries_used=False,physical_scenes=7,query_tasks=8,scenes=rows,fern_common_heldout_observations=paired,artifacts=artifacts,implementation_sha256=sha256_file(__file__),limitations=['Sparse SfM reference, existing poses; not independent dense truth','Old fern calibration may have seen audit point IDs; new local calibration excludes them','No scene dropped from the task inventory; failed controls are not zero-filled retrieval scores','No new retrieval result is inferred from geometry errors'])
out.write_text(json.dumps(result,indent=2)+'\n');print(out)
