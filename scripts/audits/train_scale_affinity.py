"""Fit structural grouping on source masks, preserving native query semantics."""
import argparse,json
from pathlib import Path
import torch
from torch import nn
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import ElementRetrievalField,load_element_retrieval_field
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--parent',default='preserved96_protected_imagecal_balanced');p.add_argument('--name',default='scale_affinity_v1');p.add_argument('--device',default='cuda:0');p.add_argument('--passes',type=int,default=12);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
 root=a.root;out=root/a.name;out.mkdir(exist_ok=False);sources={str(p):sha256_file(p) for p in [Path(__file__),Path('radio_gs/models/scale_affinity_readout.py'),Path('radio_gs/models/element_retrieval_field.py')]}
 info=json.loads((root/'teachers.json').read_text());receipt=json.loads((root/'episodes.json').read_text());assert sha256_file(root/'episodes.pt')==receipt['sha256'];data=torch.load(root/'episodes.pt',weights_only=False);assert sha256_file(root/'teachers.json')==data['teacher_manifest_sha256'];assert sha256_file(root/'responsibility.json')==info['authority_sha256'];auth=json.loads((root/'responsibility.json').read_text())
 geometry=FrozenGaussianGeometry.load(a.geometry,expected_sha256=auth['metadata']['frozen_geometry_sha256'],device=a.device);geometry.assert_binding(data['geometry_binding'])
 report=json.loads((root/a.parent/'report.json').read_text());parent,payload=load_element_retrieval_field(root/a.parent/'field.pt',expected_sha256=report['field_sha256'],geometry=geometry,expected_query_encoder=json.loads((root/'native_query_encoder.json').read_text()),device=a.device)
 views=data['proposal_views'].tolist();trainviews={info['frames'].index(f) for f in info['train_frames']};valviews={info['frames'].index(f) for f in info['validation_frames']};assert not trainviews&valviews
 rp=data['region_probability'].float().to(a.device);seen=data['seen'].to(a.device);positive=(rp>.7)&parent.observed;negative={};scales={};ids={};sizes={}
 for vi in set(views):
  region_ids=[i for i,v in enumerate(views) if v==vi];other=positive[region_ids].any(0)
  for i in region_ids:
   neg=(rp[i]<.1)&seen[vi]&parent.observed&other
   pos=torch.where(positive[i]&seen[vi])[0];neg=torch.where(neg)[0]
   if len(pos)<8 or len(neg)<8:continue
   xyz=geometry.xyz[pos];diameter=(torch.quantile(xyz,.95,dim=0)-torch.quantile(xyz,.05,dim=0)).norm().clamp_min(1e-6)
   ids[i]=(pos,neg);scales[i]=diameter; sizes[i]=len(pos)
 train=[i for i in ids if views[i] in trainviews];validation=[i for i in ids if views[i] in valviews]
 assert train and validation
 scale_grid=torch.quantile(torch.stack([scales[i] for i in train]),torch.tensor([.1,.3,.5,.7,.9],device=a.device)).tolist()
 architecture=dict(payload['architecture'],scale_affinity={'scales':scale_grid,'default_scale':2})
 field=ElementRetrievalField(**architecture).to(a.device);missing=field.load_state_dict(payload['state_dict'],strict=False);assert not missing.unexpected_keys and all(k.startswith('affinity.') for k in missing.missing_keys)
 field.requires_grad_(False);field.affinity.requires_grad_(True);field.affinity.bind_geometry(geometry.xyz,payload['query_context']['prompt_world_normalization'])
 queries=data['queries']['prompt'].float().to(a.device)
 with torch.no_grad():seed_rows=torch.cat([field.affinity.seed_rows(q) for q in queries.split(16)])
 admitted={};rejected=[]
 for i in ids:
  pos,neg=seed_rows[i].tolist()
  if pos==neg or not parent.observed[pos] or not parent.observed[neg] or not positive[i,pos] or rp[i,neg]>.1:rejected.append(i)
  else:admitted[i]=(pos,neg)
 train=[i for i in train if i in admitted];validation=[i for i in validation if i in admitted];assert train and validation
 structure=nn.Parameter(parent.latent[:,64:].detach().clone());initial=structure.detach().clone();opt=torch.optim.Adam([structure,*field.affinity.parameters()],lr=.01)
 @torch.no_grad()
 def validate():
  byscale=[]
  for scale in field.affinity.scales:
   results=[]
   for i in validation:
    p,n=admitted[i];pos,neg=ids[i];rows=torch.cat([pos,neg]);target=torch.cat([torch.ones(len(pos),device=a.device),torch.zeros(len(neg),device=a.device)]).bool()
    pred=field.affinity.logits_at_scale(structure,p,n,scale,rows)>0;results.append(float((pred&target).sum()/(pred|target).sum().clamp_min(1)))
   byscale.append(sum(results)/len(results))
  return byscale
 before=validate();direct_state={k:v.detach().cpu().clone() for k,v in field.state_dict().items()};direct_state['affinity.default_scale'].fill_(max(range(len(before)),key=lambda i:before[i]));counts=torch.zeros(len(structure),device=a.device,dtype=torch.long);history=[]
 for epoch in range(a.passes):
  order=torch.randperm(len(train)).tolist();losses=[]
  for j in order:
   i=train[j];pos,neg=ids[i];p,n=admitted[i]
   pp=pos[torch.randint(len(pos),(256,),device=a.device)];nnn=neg[torch.randint(len(neg),(256,),device=a.device)];rows=torch.cat([pp,nnn]);target=torch.cat([torch.ones(len(pp),device=a.device),torch.zeros(len(nnn),device=a.device)])
   logits=field.affinity.logits_at_scale(structure,p,n,scales[i],rows)
   loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
   opt.zero_grad();loss.backward();opt.step();counts[rows.unique()]+=1;counts[p]+=1;counts[n]+=1;losses.append(float(loss.detach()))
  row={'epoch':epoch+1,'balanced_source_pair_bce':sum(losses)/len(losses),'updated_elements':int((counts>0).sum())};history.append(row);print(json.dumps(row),flush=True)
 after=validate();selected=max(range(len(after)),key=lambda i:after[i]);field.affinity.default_scale.fill_(selected)
 with torch.no_grad():field.latent[:,64:].copy_(structure)
 assert torch.equal(field.latent[:,:64],parent.latent[:,:64]);assert torch.equal(field.native_decoder.weight,parent.native_decoder.weight)
 def save(name,state,diagnostic):
  destination=root/name
  if destination!=out:destination.mkdir(exist_ok=False)
  torch.save(dict(payload,architecture=architecture,state_dict=state),destination/'field.pt')
  cold,_=load_element_retrieval_field(destination/'field.pt',expected_sha256=sha256_file(destination/'field.pt'),geometry=geometry,expected_query_encoder=payload['query_encoder'],device=a.device)
  with torch.no_grad():
   native_error=max(float((cold.retrieve(data['queries'][m][:3].float().to(a.device),m)['scores']-parent.retrieve(data['queries'][m][:3].float().to(a.device),m)['scores']).abs().max()) for m in ['text','image'])
  assert native_error==0
  record={'schema':'scale_affinity_source_training.v1','field_sha256':sha256_file(destination/'field.pt'),'parent_field_sha256':report['field_sha256'],'episode_sha256':receipt['sha256'],'source_only':True,'benchmark_data_used':False,'native_text_image_max_error':native_error,'native_prefix_exactly_preserved':True,'source_training_regions':train,'source_validation_regions':validation,'rejected_seed_regions':rejected,'scale_grid':scale_grid,'default_scale_index':int(state['affinity.default_scale']),'selection_policy':'maximum mean source-heldout known-element IoU; no benchmark labels','validation_before':before,'validation_after':after,'structural_training':not diagnostic,'structural_updated_rows':int((counts>0).sum()) if not diagnostic else 0,'structural_change_norm':float((structure-initial).norm()) if not diagnostic else 0,'history':history if not diagnostic else [],'implementation_sources':sources,'grounded_text_supervision_sha256':report['grounded_text_supervision_sha256'],'geometry_unchanged':True,'query_contract':'same registered positive/negative world coordinates from prompt; DINO portion unused by this readout; scale chosen globally on source validation'}
  (destination/'report.json').write_text(json.dumps(record,indent=2)+'\n')
 for p,digest in sources.items():assert sha256_file(Path(p))==digest,'implementation changed during training'
 save(a.name,{k:v.detach().cpu() for k,v in field.state_dict().items()},False);save(a.name+'_direct_control',direct_state,True)
 print(json.dumps({'output':str(out),'source_validation_before':before,'after':after}),flush=True)
if __name__=='__main__':main()
