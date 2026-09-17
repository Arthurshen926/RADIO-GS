"""Fit image-query-specific native boundaries from consistent source matches.

Native directions and scores are frozen. Target-view misses are not negatives;
only the scope of an explicitly matched source region supplies mask evidence.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_threshold import fit_threshold_ridge
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
 base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=base/'preserved_source_inventory_20260916'/a.scene
 out=root/'native_image_query_threshold96';out.mkdir(exist_ok=False)
 fp=root/'exact_render_query_threshold96/field.pt';assert sha256_file(fp)==json.loads((fp.parent/'report.json').read_text())['field_sha256'];payload=torch.load(fp,weights_only=False)
 ep=base/'scope_consistent_episodes_20260916'/a.scene/'episodes.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];data=torch.load(ep,weights_only=False)
 info=json.loads((root/'teachers.json').read_text());assert sha256_file(root/'teachers.json')==data['teacher_manifest_sha256'];assert data['geometry_binding']==payload['geometry_binding']
 field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
 probability=data['region_probability'].float().to(a.device);seen=data['seen'].to(a.device)
 tasks=[e for e in data['episodes'] if e['modality']=='image' and e['split']=='train']
 ids=sorted({e['query_region'] for e in tasks});q=data['queries']['image'][ids].float().to(a.device)
 scores=field.native_cosine(q,modality="image");scale=field.readout_log_scale['image'].clamp(-4,6).exp()
 accepted=[];thresholds=[];records=[]
 for col,ri in enumerate(ids):
  targets=[e['target_region'] for e in tasks if e['query_region']==ri]
  pos=torch.zeros(len(field.latent),device=a.device,dtype=torch.bool);neg=pos.clone()
  for target in targets:
   vi=int(data['proposal_views'][target]);assert info['frames'][vi] in info['train_frames']
   pos|=(probability[target]>.7)&seen[vi];neg|=(probability[target]<.1)&seen[vi]
  assert info['frames'][int(data['proposal_views'][ri])] in info['train_frames']
  known=(pos^neg)&field.observed;ps=scores[known&pos,col];ns=scores[known&neg,col]
  if min(len(ps),len(ns))<20:continue
  ranking=binary_ranking_metrics(scores[known,col],pos[known])
  if ranking['auroc']<=.5:continue
  lo,hi=-1.,1.
  for _ in range(32):
   mid=(lo+hi)/2;mass=((scale*(ps-mid)).sigmoid().mean()+(scale*(ns-mid)).sigmoid().mean())/2
   if mass>.5:lo=mid
   else:hi=mid
  cutoff=(lo+hi)/2;accepted.append(col);thresholds.append(cutoff)
  records.append(dict(query_region=ri,query_view=int(data['proposal_views'][ri]),target_regions=targets,threshold=cutoff,positive_elements=len(ps),negative_elements=len(ns),source_auroc=ranking['auroc']))
 features=q[accepted];target=torch.tensor(thresholds,device=a.device);assert len(target)>=4
 groups=torch.tensor([r['query_view'] for r in records],device=a.device);assert len(groups.unique())>=3
 penalties=[.01,.1,1.,10.];errors=[]
 for penalty in penalties:
  losses=[]
  for view in groups.unique():
   keep=groups!=view;w,b=fit_threshold_ridge(features[keep],target[keep],penalty)
   losses.append(float(((F.normalize(features[~keep],dim=-1)@w+b)-target[~keep]).square().mean()))
  errors.append(sum(losses)/len(losses))
 chosen=min(range(len(penalties)),key=errors.__getitem__);w,b=fit_threshold_ridge(features,target,penalties[chosen])
 config={'lower':float(target.min()),'upper':float(target.max())}
 architecture=dict(payload['architecture'],native_query_thresholds=dict(payload['architecture'].get('native_query_thresholds',{}),image=config))
 state=dict(payload['state_dict']);state.update({'native_query_thresholds.image.linear.weight':w.cpu()[None],'native_query_thresholds.image.linear.bias':b.cpu()[None],'native_query_thresholds.image.bounds':torch.tensor([config['lower'],config['upper']])})
 cold=ElementRetrievalField(**architecture);cold.load_state_dict(state,strict=True)
 for k,v in payload['state_dict'].items():assert torch.equal(v,cold.state_dict()[k]),k
 torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
 report=dict(schema='native_image_query_threshold.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=sha256_file(fp),source_only=True,benchmark_data_used=False,
  episode_sha256=sha256_file(ep),all_parent_tensors_unchanged=True,native_cosine_unchanged=True,text_prompt_unchanged=True,
  supervision='matched source region full-scope pseudo mask: >.7 positive, <.1 visible negative; contradictory multi-view rows unknown; no absent-view negatives',
  validation='leave-query-view-out ridge prediction of source cutoff; geometry/target evidence are not independent heldout',
  query_fits=records,ridge_candidates=penalties,ridge_group_mse=errors,selected_ridge=penalties[chosen],bounds=config,
  implementation_sha256=sha256_file(__file__),module_sha256=sha256_file('radio_gs/models/native_query_threshold.py'))
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,'queries',len(records),'bounds',config,'ridge',penalties[chosen],flush=True)

if __name__=='__main__':
 with torch.no_grad():main()
