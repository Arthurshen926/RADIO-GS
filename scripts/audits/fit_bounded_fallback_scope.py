"""Fit a finite fallback structural logit bound using source masks only."""
import argparse,json,hashlib,shutil
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.bounded_scope_evidence import bounded_scope_logits
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--withheld-concepts',action='store_true');p.add_argument('--positive-only',action='store_true');a=p.parse_args();torch.set_num_threads(4)
 out=a.root/('positive_fallback_source_control' if a.positive_only else 'bounded_fallback_withheld' if a.withheld_concepts else 'bounded_fallback_scope');out.mkdir(exist_ok=False);snap=out/'sources';snap.mkdir()
 implementations={}
 for f in [Path(__file__),Path('radio_gs/models/element_retrieval_field.py'),Path('radio_gs/models/bounded_scope_evidence.py')]:
  shutil.copy2(f,snap/f.name);implementations[str(f)]=sha256_file(f)
 fp=a.root/'joint_supervised_image_residual/field.pt';pr=json.loads(fp.with_name('report.json').read_text());assert sha256_file(fp)==pr['field_sha256'];payload=torch.load(fp,weights_only=False)
 model=ElementRetrievalField(**payload['architecture']).to(a.device);model.load_state_dict(payload['state_dict']);model.requires_grad_(False)
 assert model.extent_benefit_admission is None
 if a.withheld_concepts:
  with torch.no_grad():model.grounded_transfer.acceptance.fill_(2.)
 ep=a.root/'caption_region_evidence.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];e=torch.load(ep,weights_only=False)
 assert e['source_only'] and not e['benchmark_data_used'] and e['geometry_binding']==payload['geometry_binding']
 info=json.loads((a.root/'teachers.json').read_text());assert e['teacher_manifest_sha256']==sha256_file(a.root/'teachers.json')
 auth=json.loads((a.root/'responsibility.json').read_text());assert sha256_file(a.root/'responsibility.json')==info['authority_sha256']
 frames=info['train_frames'];assert len(frames)>=8;validation=frames[-4:];train=frames[:-4];views=[]
 for frame in frames:
  targets={}
  for i,r in enumerate(e['records']):
   if r['frame']==frame and r['confidence']>=.7 and int(e['masks'][i].sum())>=20:
    for qi in r['phrase_indices']:targets[qi]=targets.get(qi,torch.zeros_like(e['masks'][i],dtype=torch.bool))|e['masks'][i].bool()
  if not targets:continue
  rec=auth['views'][info['frames'].index(frame)];hp=a.root/rec['relative_path'];assert sha256_file(hp)==rec['sha256'];hit=torch.load(hp,map_location=a.device,weights_only=False)
  gid,pid,w=hit['gaussian_ids'].long(),hit['pixel_ids'].long(),hit['base_weights'].float();pixels=e['masks'][0].numel();mass=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,w);use=model.observed[gid];gid,pid,w=gid[use],pid[use],w[use];visible=torch.zeros_like(mass).scatter_add_(0,pid,w)
  mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/visible[pid].clamp_min(1e-8),(pixels,len(model.latent))).coalesce()
  views.append(dict(frame=frame,targets=targets,mat=mat,valid=(mass>1e-8)&(visible/mass.clamp_min(1e-8)>.5)))
 strengths=[0.,.5,1.,2.,4.];observations=[];captured={}
 def hook(module,inputs,output):captured['accepted']=output[1]
 handle=model.grounded_transfer.register_forward_hook(hook)
 with torch.no_grad():
  for start in range(0,len(e['text_embeddings']),8):
   q=e['text_embeddings'][start:start+8].float().to(a.device);parent=model.logits(q,'text');_,concept,_=model.grounded_transfer.route(q);active=captured['accepted']&~concept
   if not active.any():continue
   native=model.readout_log_scale['text'].clamp(-4,6).exp()*(model.native_cosine(q)-model.native_query_thresholds['text'](q)[None])
   for v in views:
    ids=[j for j in range(len(q)) if bool(active[j]) and start+j in v['targets']]
    if not ids:continue
    target=torch.stack([v['targets'][start+j].flatten() for j in ids],1).to(a.device)
    columns=[parent[:,ids].sigmoid()]+[(native[:,ids]+b*(parent[:,ids]>0)).sigmoid() if a.positive_only else bounded_scope_logits(native[:,ids],parent[:,ids]>0,b).sigmoid() for b in strengths]
    pred=(torch.sparse.mm(v['mat'],torch.cat(columns,1)*model.observed[:,None])>.5)&v['valid'][:,None]
    scores=[]
    for k in range(len(columns)):
     x=pred[:,k*len(ids):(k+1)*len(ids)];scores.append(((x&target).sum(0)/(x|target).sum(0).clamp_min(1)).tolist())
    for j,local in enumerate(ids):
     mask=v['targets'][start+local];scope=hashlib.sha256(mask.numpy().tobytes()).hexdigest()
     observations.append(dict(frame=v['frame'],query_index=start+local,phrase=e['phrases'][start+local],scope=scope,iou=[scores[k][j] for k in range(len(columns))]))
   print('queries',start,'fallback observations',len(observations),flush=True)
 handle.remove()
 def aggregate(frames):
  grouped={}
  for r in observations:
   if r['frame'] in frames:grouped.setdefault((r['frame'],r['scope']),[]).append(torch.tensor(r['iou']))
  if not grouped:raise ValueError('No fallback scope observations in source split')
  return torch.stack([torch.stack(x).mean(0) for x in grouped.values()]).mean(0).tolist(),len(grouped)
 training,nt=aggregate(train);heldout,nv=aggregate(validation);index=max(range(1,len(training)),key=lambda k:training[k]);strength=strengths[index-1]
 selection=dict(payload['architecture']['grounded_transfer']['identity_selection'],completion_mode='native_bounded_extent',scope_logit_bound=strength)
 arch=dict(payload['architecture'],grounded_transfer=dict(payload['architecture']['grounded_transfer'],identity_selection=selection))
 cold=ElementRetrievalField(**arch);cold.load_state_dict(payload['state_dict'],strict=True)
 if not a.positive_only:torch.save(dict(payload,architecture=arch),out/'field.pt')
 report=dict(schema='bounded_fallback_scope.v1',positive_only_diagnostic=a.positive_only,field_sha256=sha256_file(out/'field.pt') if not a.positive_only else None,parent_field_sha256=sha256_file(fp),source_only=True,benchmark_data_used=False,geometry_unchanged=True,all_parent_tensors_exact=True,source_concept_matches_withheld=a.withheld_concepts,deployment_lexical_routes_unchanged=True,selected_strength=strength,strengths=strengths,score_order=['retained']+strengths,train_iou=training,validation_iou=heldout,training_frames=train,validation_frames=validation,unique_scopes=[nt,nv],observations=observations,source_acceptance=heldout[index]>=heldout[0],selection='Best training unique-scope macro; four source views validate only, no benchmark tuning; parent saw these source views, not independent family validation',evidence_sha256=sha256_file(ep),implementations=implementations)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('selected',strength,'train',training,'validation',heldout,flush=True)

if __name__=='__main__':main()
