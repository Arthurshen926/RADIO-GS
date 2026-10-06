"""Fit scope IoU and ranking to source renders, without changing teacher features."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.scope_quality import ScopeQuality,scope_features


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
 out=Path('output/scope_complete_next36')/a.key;data=load(out/'field.pt');root=Path(data['root']);graph=data['scope_hypergraph'];records=json.loads((out/'construction.json').read_text())['records']
 parent=load(data['base_field']);model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.requires_grad_(False);model.eval()
 info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());ep=load(root/'episodes.pt');teacher=load(root/info['teacher_files']['siglip']['path']);masks=ep['masks'];views=ep['proposal_views'];descriptors=teacher['descriptors'].float();raw=load(root/'source_grounded_text.pt')
 extra=Path('output/scope_hierarchy_full_next33')/a.key/'small_teacher.pt'
 if not extra.exists():extra=Path('output/small_region_restoration_next32')/a.key/'small_teacher.pt'
 if extra.exists():
  small=load(extra);masks=torch.cat([masks,small['masks']]);views=torch.cat([views,small['proposal_views']]);descriptors=torch.cat([descriptors,small['descriptors'].float()])
 n,hypotheses=graph['shape'];scope=torch.sparse_coo_tensor(graph['indices'].cuda(),graph['weights'].cuda(),graph['shape']).to_dense();pixels=masks.shape[-2]*masks.shape[-1];sets={'train':[],'validation':[]}
 with torch.no_grad():
  for vi,frame in enumerate(info['frames']):
   hit=load(root/auth['views'][vi]['relative_path']);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();w=hit['base_weights'].float().cuda();total=torch.zeros(pixels,device='cuda').scatter_add_(0,pid,w)
   ids=torch.where(views==vi)[0];ids=ids[torch.linspace(0,len(ids)-1,min(32,len(ids))).long()];queries=[('image',descriptors[ids].cuda(),masks[ids].flatten(1).cuda())]
   if frame in raw['frames']:
    j=raw['frames'].index(frame);eligible=(raw['scores'][j]>=.7)&((raw['probability'][j]>.5).flatten(1).sum(1)>=20);qi=torch.where(eligible)[0]
    if len(qi):queries.append(('text',ep['queries']['text'][qi].float().cuda(),(raw['probability'][j,qi]>.5).flatten(1).cuda()))
   # A query crop must not already be one of the candidate's prototype views.
   independent=torch.tensor([vi not in r['views'] for r in records],device='cuda')
   for modality,q,target in queries:
    obs=model.observed_for(modality);ow=w*obs[gid];mass=torch.zeros_like(total).scatter_add_(0,pid,ow);operator=torch.sparse_coo_tensor(torch.stack([pid,gid]),ow/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce();valid=(mass/total.clamp_min(1e-8)>.5)&(total>1e-8)
    projected=(torch.sparse.mm(operator,scope)>.5)&valid[:,None];area=projected.sum(0);intersection=target.float()@projected.float();label=intersection/(target.sum(1)[:,None]+area[None]-intersection).clamp_min(1)
    available=(area>=8)&independent
    if not available.any():continue
    features=[]
    for start in range(0,len(q),16):features.append(scope_features(model,graph,q[start:start+16],modality))
    x=torch.cat(features);group='validation' if frame in info['validation_frames'] else 'train';sets[group].append((x.cpu(),label.cpu(),available.cpu()[None].expand(len(q),-1),modality))
   print(a.key,'source quality targets',frame,flush=True)
 del scope,model
 x,y,valid=[torch.cat([row[i] for row in sets['train']]).cuda() for i in range(3)];vx,vy,vvalid=[torch.cat([row[i] for row in sets['validation']]).cuda() for i in range(3)]
 mean=x[valid].mean(0);std=x[valid].std(0).clamp_min(.01);head=ScopeQuality(mean,std).cuda();opt=torch.optim.Adam(head.parameters(),lr=.003)
 def measure(xx,yy,vv):
  with torch.no_grad():
   native=xx[:,:,0].masked_fill(~vv,-1e4).argmax(1);learned=head(xx).masked_fill(~vv,-1e4).argmax(1);r=torch.arange(len(xx),device='cuda')
   return dict(native_iou=float(yy[r,native].mean()),learned_iou=float(yy[r,learned].mean()),oracle_iou=float(yy.masked_fill(~vv,0).max(1).values.mean()),queries=len(xx))
 history=[dict(step=0,validation=measure(vx,vy,vvalid))]
 for step in range(600):
  ix=torch.randint(len(x),(32,),device='cuda');xx,yy,vv=x[ix],y[ix],valid[ix];z=head(xx);prediction=z.sigmoid()
  weights=torch.where(yy>=.5,5.,1.)*vv;regression=((prediction-yy).square()*weights).sum()/weights.sum().clamp_min(1)
  target=(10*yy).masked_fill(~vv,-1e4).softmax(1);rank=-(target*z.masked_fill(~vv,-1e4).log_softmax(1)).sum(1);has_target=yy.masked_fill(~vv,0).max(1).values>=.25
  loss=regression+.2*(rank*has_target).sum()/has_target.sum().clamp_min(1)
  opt.zero_grad();loss.backward();opt.step()
  if (step+1)%100==0:
   row=dict(step=step+1,loss=float(loss.detach()),validation=measure(vx,vy,vvalid));history.append(row);print(a.key,row,flush=True)
 torch.save(dict(state=head.state_dict(),mean=mean.cpu(),std=std.cpu()),out/'scope_quality.pt');report=dict(key=a.key,train_queries=len(x),validation_queries=len(vx),native_coordinates_frozen=True,benchmark_supervision=False,history=history)
 (out/'scope_quality_training.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
