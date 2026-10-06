"""Source-only query collision audit; disagreement is not human scope truth."""
import argparse,json,hashlib
from pathlib import Path
from itertools import combinations
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--query-relative-scope',action='store_true');a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';out=Path('output/current_recipe_audit_20260918')/a.scene/('source_collisions_relative.json' if a.query_relative_scope else 'source_collisions_detailed.json');assert not out.exists()
fp=root/'counterfactual_extent_benefit_admission/field.pt';assert sha256_file(fp)==json.loads(fp.with_name('report.json').read_text())['field_sha256'];payload=torch.load(fp,weights_only=False,map_location='cpu');field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
if a.query_relative_scope:
 from radio_gs.querying.query_relative_scope import install_query_relative_scope_control
 install_query_relative_scope_control(field)
ep=root/'caption_region_evidence.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];e=torch.load(ep,weights_only=False);assert e['source_only'] and not e['benchmark_data_used'];assert e['geometry_binding']==payload['geometry_binding'];masks={}
for ri,r in enumerate(e['records']):
 if r['confidence']<.7 or int(e['masks'][ri].sum())<20:continue
 for qi in set(r['phrase_indices']):
  key=(qi,r['frame']);masks[key]=masks.get(key,torch.zeros_like(e['masks'][ri],dtype=torch.bool))|e['masks'][ri].bool()
ids=sorted({qi for qi,frame in masks});rows=[]
with torch.no_grad():
 for start in range(0,len(ids),8):
  qi=ids[start:start+8];q=e['text_embeddings'][qi].float().to(a.device);index,concept,similarity=field.grounded_transfer.route(q);capture={}
  def hh(module,inputs,output):capture['eligible']=output[1].tolist()
  def gh(module,inputs,output):capture['allowed']=output.tolist()
  h=field.grounded_transfer.register_forward_hook(hh);g=field.extent_benefit_admission.register_forward_hook(gh)
  try:scores=field.retrieve(q,'text')['scores']
  finally:h.remove();g.remove()
  for j,i in enumerate(qi):
   binary=(scores[:,j]>.5).cpu().numpy().tobytes();rows.append(dict(query=i,phrase=e['phrases'][i],concept=int(index[j]),concept_accepted=bool(concept[j]),scope_used=capture['eligible'][j] and capture['allowed'][j],foreground_elements=int((scores[:,j]>.5).sum()),score_sha256=hashlib.sha256(scores[:,j].contiguous().cpu().numpy().tobytes()).hexdigest(),membership_sha256=hashlib.sha256(binary).hexdigest()))
  print(a.scene,start,len(ids),flush=True)
pairs=[]
for left,right in combinations(rows,2):
 if left['membership_sha256']!=right['membership_sha256']:continue
 frames=sorted({f for qi,f in masks if qi==left['query']} & {f for qi,f in masks if qi==right['query']});comparisons=[]
 for frame in frames:
  x=masks[left['query'],frame];y=masks[right['query'],frame];comparisons.append(dict(frame=frame,iou=float((x&y).sum()/(x|y).sum().clamp_min(1)),left_pixels=int(x.sum()),right_pixels=int(y.sum())))
 if comparisons:pairs.append(dict(left=left,right=right,coobserved=comparisons,mean_source_mask_iou=sum(r['iou'] for r in comparisons)/len(comparisons),minimum_source_mask_iou=min(r['iou'] for r in comparisons)))
report=dict(schema='source_query_collision_audit.v2',query_relative_scope_control=a.query_relative_scope,source_only=True,benchmark_data_used=False,verified_physical_identity=False,human_scope_labels=False,field_sha256=sha256_file(fp),evidence_sha256=sha256_file(ep),implementation_sha256=sha256_file(__file__),query_count=len(rows),queries=rows,identical_membership_coobserved_pairs=pairs,low_agreement_pairs=sum(r['minimum_source_mask_iou']<.5 for r in pairs),limitation='Exact binary element membership equality, not equality of rendered probabilities. Source mask disagreement flags cases to review, not proof of distinct physical or intended query scope. Alias-rich v1 native cache remains in use.')
out.write_text(json.dumps(report,indent=2)+'\n');print(a.scene,'coobserved identical pairs',len(pairs),'low agreement',report['low_agreement_pairs'])
