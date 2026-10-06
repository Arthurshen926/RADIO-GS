"""Seal two bounded source-selected controls and every evaluated output contract."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 results={};bindings={}
 for scene in ['figurines','ramen']:
  base=Path('output/depth_opacity_pipeline_20260918')/scene;field=base/'reconstruction/field'
  parent=torch.load(field/'joint_supervised_image_residual/field.pt',weights_only=False)
  old={t:json.loads((base/(t+'_online.json')).read_text())['runs']['counterfactual_extent_benefit_admission'] for t in ['text','reference']}
  row=dict(before=dict(text_iou=old['text']['observation_miou'],text_ap=old['text']['average_precision_macro'],selected_iou=json.loads((base/'selected_output/report.json').read_text())['mean_iou']))
  for variant in ['bounded','positive']:
   root=Path('output/'+variant+'_fallback_repair_20260918')/scene;run=variant+'_fallback_admission'
   execution=json.loads((root/'execution.json').read_text());assert execution['complete']
   for stage in execution['stages']:
    assert stage['status']=='complete';path=root/(stage['task']+'_'+stage['mode']+'.json');assert sha256_file(path)==stage['output_sha256']
   model=torch.load(field/run/'field.pt',weights_only=False);assert model['geometry_binding']==parent['geometry_binding']
   assert all(torch.equal(v,model['state_dict'][k]) for k,v in parent['state_dict'].items())
   assert sha256_file(field/run/'field.pt')==execution['field_sha256']
   scores={}
   for task in ['text','reference']:
    online=json.loads((root/(task+'_online.json')).read_text());replay=json.loads((root/(task+'_replay.json')).read_text());assert online['runs']==replay['runs']
    current=online['runs'][run]
    if task=='text':
     assert [(x['frame'],x['category']) for x in current['observations']]==[(x['frame'],x['category']) for x in old[task]['observations']]
     scores.update(text_iou=current['observation_miou'],text_ap=current['average_precision_macro'],categories=current['per_category_iou'])
    else:
     assert len(current['observations'])==len(old[task]['observations'])
     for a,b in zip(current['observations'],old[task]['observations']):
      assert all(a[k]==b[k] for k in ['reference_frame','target_frame','category','prompt_available','image','prompt'])
     scores['image_prompt_exact']=True;scores['reference']=current['macro']
    for name in [task+'_online.json',task+'_replay.json',task+'_online.routes.json']:
     p=root/name;bindings[str(p)]=sha256_file(p)
   selected=json.loads((root/'selected_output/report.json').read_text());scores['selected_iou']=selected['mean_iou']
   for p in [root/'execution.json',root/'selected_output/report.json',field/run/'field.pt',field/run/'report.json']:bindings[str(p)]=sha256_file(p)
   row[variant]=scores
  replay=json.loads((Path('output/positive_fallback_repair_20260918')/scene/'legacy_replay.json').read_text())['runs']['counterfactual_extent_benefit_admission'];assert replay==old['text']
  row['legacy_text_replay_exact']=True;results[scene]=row
 out=Path('paper/artifacts/fallback_scope_repair_20260918.json');assert not out.exists()
 out.write_text(json.dumps(dict(independent_test=False,global_promotion=False,selected_gt_released_jpeg_equivalence_verified=False,results=results,bindings=bindings,implementation_sha256=sha256_file(__file__)),indent=2)+'\n')
 print(json.dumps({s:{k:{m:v[m] for m in ['text_iou','text_ap','selected_iou']} for k,v in r.items() if isinstance(v,dict)} for s,r in results.items()},indent=2))

if __name__=='__main__':main()
