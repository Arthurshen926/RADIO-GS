"""Source-only regularized native reprojection; no structural loss or query adaptation."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(4)
out=Path('output/native_reprojection_system_20260919')/a.scene;out.mkdir(parents=True,exist_ok=False)
reg=dict(scene=a.scene,source_only=True,geometry_fixed=True,decoder_fixed=True,observed_fixed=True,other_query_heads_rebuilt=False,native_classifier_only=True,ridge_lambda=.1,max_relative_node_update=.1,cg_iterations=32,fit='first12 original source train views',update_validation='last4 original train views; baseline initialization saw these views, not independent holdout',no_target_tuning=True,implementation_sha256=sha256_file(__file__))
(out/'registration.json').write_text(json.dumps(reg,indent=2)+'\n')
def deny(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  s=str(args[0])
  if '.labels.ply' in s or 'eval_gt' in s or 'official_protocol_evaluation' in s:raise RuntimeError('No benchmark or evaluation reads during fitting')
sys.addaudithook(deny)
def load(path,h=None):
 if h:assert sha256_file(path)==h,path
 return torch.load(path,map_location='cpu',weights_only=False)
r=Path('output/scannet_frozen_field_20260918')/a.scene/'reconstruction/field';info=json.loads((r/'teachers.json').read_text());ep=load(r/'episodes.pt',json.loads((r/'episodes.json').read_text())['sha256']);auth=json.loads((r/'responsibility.json').read_text());assert sha256_file(r/'responsibility.json')==info['authority_sha256'];parent=r/'pixel_evidence96';payload=load(parent/'field.pt',json.loads((parent/'report.json').read_text())['field_sha256']);sd=payload['state_dict'];basis=sd['native_decoder.weight'].float();observed=sd['observed'].bool();z0=sd['latent'][:,:64].float().to(a.device);n=len(z0)
rec=info['teacher_files']['siglip'];desc=F.normalize(load(r/rec['path'],rec['sha256'])['descriptors'].float(),dim=1)@basis
rows=[]
for frame in info['train_frames']:
 vi=info['frames'].index(frame);ids=torch.where(ep['proposal_views']==vi)[0];mask=ep['masks'][ids].flatten(1).float();count=mask.sum(0);hitrec=auth['views'][vi];hit=load(r/hitrec['relative_path'],hitrec['sha256']);pix=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();w=hit['base_weights'].float();keep=observed[gid];pix,gid,w=pix[keep],gid[keep],w[keep];mass=torch.zeros(mask.shape[1]).scatter_add_(0,pix,w);valid=(count>0)&(mass>1e-8);keep=valid[pix];pix,gid,w=pix[keep],gid[keep],w[keep];remap=valid.long().cumsum(0)-1;values=w/mass[pix].clamp_min(1e-8);idx=torch.stack([remap[pix],gid]);target=(mask[:,valid].T@desc[ids])/count[valid,None]
 rows.append((frame,idx,values,target));print(a.scene,'source',frame,int(valid.sum()),flush=True)
def assemble(items):
 offset=0;idx=[];val=[];targets=[]
 for _,ij,w,y in items:
  ij=ij.clone();ij[0]+=offset;idx.append(ij);val.append(w);targets.append(y);offset+=len(y)
 indices=torch.cat(idx,1).to(a.device);values=torch.cat(val).to(a.device);A=torch.sparse_coo_tensor(indices,values,(offset,n)).coalesce();return A,A.transpose(0,1).coalesce(),torch.cat(targets).to(a.device)
A,AT,Y=assemble(rows[:12]);V,VT,T=assemble(rows[12:]);assert len(rows)==16 and len(rows[12:])==4
mass=torch.zeros(n,device=a.device).scatter_add_(0,A.indices()[1],A.values());ridge=.1*mass;diag=torch.zeros(n,device=a.device).scatter_add_(0,A.indices()[1],A.values().square())+ridge;diag=diag.clamp_min(1e-8)
def mm(M,x):return torch.sparse.mm(M,x)
def op(x):return mm(AT,mm(A,x))+ridge[:,None]*x
b=mm(AT,Y-mm(A,z0));delta=torch.zeros_like(z0);res=b.clone();pre=res/diag[:,None];direction=pre.clone();rz=(res*pre).sum(0);initial=res.square().sum(0).sqrt();history=[];start=time.time()
for i in range(32):
 hd=op(direction);alpha=rz/(direction*hd).sum(0).clamp_min(1e-20);delta+=direction*alpha;res-=hd*alpha;pre=res/diag[:,None];new=(res*pre).sum(0);direction=pre+direction*(new/rz.clamp_min(1e-20));rz=new
 if i%8==7:history.append(dict(iteration=i+1,relative_residual=float((res.square().sum(0).sqrt()/initial.clamp_min(1e-8)).max())));print(a.scene,history[-1],flush=True)
bound=.1*z0.norm(dim=1,keepdim=True);delta*=torch.minimum(torch.ones_like(bound),bound/delta.norm(dim=1,keepdim=True).clamp_min(1e-12));delta[~observed.to(a.device)]=0;z=z0+delta
metrics={}
for label,W,y in [('fit',A,Y),('update_validation',V,T)]:
 metrics[label]={}
 for name,x in [('baseline',z0),('candidate',z)]:
  pred=mm(W,x);metrics[label][name]=dict(mse=float((pred-y).square().mean()),cosine=float(F.cosine_similarity(pred,y,dim=1).mean()))
assert torch.equal(z[~observed.to(a.device)],z0[~observed.to(a.device)])
field=out/'native_field.pt';torch.save(dict(native_latent=z.cpu(),native_decoder=basis,observed=observed,geometry_binding=payload['geometry_binding']),field)
report=dict(reg,metrics=metrics,source_acceptance=metrics['update_validation']['candidate']['mse']<=metrics['update_validation']['baseline']['mse'],cg_history=history,solver_seconds=time.time()-start,geometry_binding=payload['geometry_binding'],parent_sha256=sha256_file(parent/'field.pt'),teacher_sha256=rec['sha256'],field_sha256=sha256_file(field),relative_update_max=float((delta.norm(dim=1)/z0.norm(dim=1).clamp_min(1e-8)).max()))
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,metrics,flush=True)
