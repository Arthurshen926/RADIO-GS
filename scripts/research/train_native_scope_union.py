"""Calibrate composed native scope responses against source text masks."""
import argparse,json,math
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.native_scope_union import NativeScopeUnion


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--source',default='output/rendered_scopes_next37');p.add_argument('--output',default='output/native_scope_union_next37');p.add_argument('--steps',type=int,default=160);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
 source=Path(a.source)/a.key;data=load(source/'field.pt');root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());ep=load(root/'episodes.pt');raw=load(root/'source_grounded_text.pt');parent=load(data['base_field']);model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.requires_grad_(False);model.eval();field=NativeScopeUnion(data['scope_hypergraph']);n=field.shape[0]
 examples=[]
 for frame in info['train_frames']:
  if frame not in raw['frames']:continue
  vi=info['frames'].index(frame);j=raw['frames'].index(frame);target=raw['probability'][j];eligible=(raw['scores'][j]>=.7)&((target>.5).flatten(1).sum(1)>=20);qi=torch.where(eligible)[0]
  if not len(qi):continue
  hit=load(root/auth['views'][vi]['relative_path']);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();weight=hit['base_weights'].float().cuda()*model.observed_for('text')[gid];pixels=target.shape[-2]*target.shape[-1];mass=torch.zeros(pixels,device='cuda').scatter_add_(0,pid,weight);op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce()
  q=ep['queries']['text'][qi].float().cuda()
  with torch.no_grad():base=model.retrieve(q,'text')['scores'].detach()
  region_cover=ep['masks'][ep['proposal_views']==vi].flatten(1).any(0).cuda()
  examples.append((op,mass>1e-8,q,target[qi].flatten(1).T.cuda(),base,region_cover))
 log_scale=torch.nn.Parameter(torch.tensor(math.log(20.),device='cuda'));threshold=torch.nn.Parameter(torch.tensor(.15,device='cuda'));opt=torch.optim.Adam([log_scale,threshold],lr=.01);history=[]
 for step in range(a.steps):
  op,valid,q,target,base,region_cover=examples[step%len(examples)];j=step%q.shape[0];q=q[j:j+1];target=target[:,j:j+1];base=base[:,j:j+1]
  response=field.response(q,log_scale,threshold)+(1-field.coverage[:,None])*base
  pred=torch.sparse.mm(op,response);truth=(target>.5).float();native_reference=torch.sparse.mm(op,base)
  qualified_negative=(target<=.2)&(native_reference<.1)&region_cover[:,None]
  known=valid[:,None]&((target>=.8)|qualified_negative);positive=truth*known;negative=(1-truth)*known
  # Object-balanced likelihood plus rendered soft IoU; unknown probability
  # pixels contribute neither foreground nor background supervision.
  bce=-(positive*pred.clamp_min(1e-6).log()).sum()/positive.sum().clamp_min(1)-(negative*(1-pred).clamp_min(1e-6).log()).sum()/negative.sum().clamp_min(1)
  intersection=(pred*positive).sum();union=((pred+truth-pred*truth)*known).sum();loss=bce+1-intersection/union.clamp_min(1)
  opt.zero_grad();loss.backward();opt.step()
  with torch.no_grad():log_scale.clamp_(0,math.log(100));threshold.clamp_(-1,1)
  if step%20==0:row=dict(step=step,loss=float(loss.detach()),scale=float(log_scale.detach().exp()),threshold=float(threshold.detach()));history.append(row);print(a.key,row,flush=True)
 data['native_scope_union']={'text':dict(log_scale=float(log_scale.detach()),threshold=float(threshold.detach()))};out=Path(a.output)/a.key;out.mkdir(parents=True,exist_ok=True);torch.save(data,out/'field.pt');(out/'construction.json').write_text((source/'construction.json').read_text());(out/'training.json').write_text(json.dumps(dict(history=history,source_views=len(examples),source_only=True,teacher_coordinates_frozen=True,image_scope_contract_unchanged=True),indent=2)+'\n')


if __name__=='__main__':main()
