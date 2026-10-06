"""Seal the fixed four-scene query-first region readout comparison."""
import json
from pathlib import Path
from statistics import mean
from radio_gs.utils.immutable_artifacts import sha256_file

SCENES={'figurines':56,'ramen':71,'teatime':59,'waldo_kitchen':22}
ARMS=['final','region_mean','region_gate','region_unrouted']

def main():
 bindings={};scenes={};pairs=[];all_unrouted=[];all_routed=[]
 def read(path):
  path=Path(path);bindings[str(path)]=sha256_file(path);return json.loads(path.read_text())
 for scene,n in SCENES.items():
  root=Path('output/query_region_readout_next25')/scene
  prep=read(root/'preparation.json');r=read(root/'evaluation/report.json');cal=read(root/'source_calibration.json')
  old=read(Path('output/full_field_rebuild_next19')/scene/'evaluation/report.json')
  routes=read(Path('output/caption_view_strength_next24')/scene/'evaluation/report.json')
  assert prep['complete'] and r['complete'] and r['baseline_disagreement_pixels']==0
  assert sha256_file(root/'scores.pt')==prep['scores_sha256']==r['scores_sha256']
  bindings[str(root/'scores.pt')]=prep['scores_sha256']
  assert r['preparation_sha256']==bindings[str(root/'preparation.json')]
  assert r['lerf2d']['final']['miou']==old['lerf2d']['final']['miou']
  assert r['lerf2d']['final']['ap']==old['lerf2d']['final']['macro_ap']
  assert r['lerf2d']['final']['observations']==[{k:x[k] for k in ['frame','query','iou','ap']} for x in old['lerf2d']['final']['observations']]
  assert cal['source_only'] and not cal['benchmark_labels_opened']
  admitted={x['query'] for x in routes['routes']}
  assert len(admitted)==prep['routed_queries']
  metrics={}
  for arm in ARMS:
   assert len(r['lerf2d'][arm]['observations'])==n
   assert r['selected'][arm]['count']==n and r['selected'][arm]['missing']==0
   metrics[arm]=dict(iou=r['lerf2d'][arm]['miou'],ap=r['lerf2d'][arm]['ap'],selected=r['selected'][arm]['miou'])
  baseline={(x['frame'],x['query']):x for x in r['lerf2d']['final']['observations']}
  for arm in ARMS[1:]:
   for x in r['lerf2d'][arm]['observations']:
    key=(x['frame'],x['query']);b=baseline[key]
    if arm=='region_unrouted' and x['query'] in admitted:assert x==b
    pairs.append(dict(scene=scene,arm=arm,frame=x['frame'],query=x['query'],routed=x['query'] in admitted,iou_delta=x['iou']-b['iou'],ap_delta=x['ap']-b['ap']))
  for x in r['lerf2d']['final']['observations']:
   (all_routed if x['query'] in admitted else all_unrouted).append(x['iou'])
  scenes[scene]=dict(metrics=metrics,route_unique=len(admitted),routed_observations=sum(x['query'] in admitted for x in r['lerf2d']['final']['observations']),source_support_fraction=prep['support_fraction'],source_prompted_top1=cal['positive_top1_among_all_source_phrases'],source_positive_above_element_threshold=cal['positive_above_source_element_threshold'])
 macro={arm:{k:mean(scenes[s]['metrics'][arm][k] for s in scenes) for k in ['iou','ap','selected']} for arm in ARMS}
 manifest=read('output/frozen_cross_benchmark_method_20260918/source_manifest.json')
 changed=[p for p,h in manifest.items() if sha256_file(p)!=h];assert not changed
 result=dict(complete=True,scenes=scenes,macro=macro,paired=pairs,observations_per_arm=208,unrouted_observations=len(all_unrouted),unrouted_baseline_iou=mean(all_unrouted),routed_observations=len(all_routed),routed_baseline_iou=mean(all_routed),frozen_source_count=len(manifest),frozen_source_changes=changed,default_promoted=False,independent_final_test=False,official_four_query=False,selected_is_volumetric_3d_iou=False,artifact_sha256=bindings)
 out=Path('output/query_region_readout_next25');(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(macro=macro,unrouted=len(all_unrouted)),indent=2))
if __name__=='__main__':main()
