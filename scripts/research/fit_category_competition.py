"""Fit partial-label category competition, retaining unknown source classes."""
import argparse,json,shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.scope_hypergraph import attach_scope_hypergraph
from radio_gs.querying.source_accepted_category import source_accepted_category_logits


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
 source=Path('output/scope_composition_native_next38')/a.key;data=load(source/'field.pt');root=Path(data['root']);parent=load(data['base_field']);model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.requires_grad_(False);model.eval();attach_scope_hypergraph(model,data)
 calibration=load(source/'calibration.pt');original=model.native_query_thresholds['text'].forward
 def threshold(q):
  delta=F.normalize(q,dim=-1)@calibration['weight'].cuda()+calibration['bias'].cuda()
  return original(q)+delta.clamp(-calibration['max_delta'],calibration['max_delta'])
 model.native_query_thresholds['text'].forward=threshold
 info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());ep=load(root/'episodes.pt');teacher=load(root/info['teacher_files']['siglip']['path']);ground=load(root/'source_grounded_text.pt');q=F.normalize(ep['queries']['text'].float().cuda(),dim=1)
 scope_receipt=json.loads((root/'bounded_fallback_withheld/report.json').read_text());admission=json.loads((Path(data['base_field']).parent/'report.json').read_text())
 with torch.no_grad():
  calibrated=model.readout_log_scale['text'].clamp(-4,6).exp()*(model.native_cosine(q)-model.native_query_thresholds['text'](q))
  logits=source_accepted_category_logits(calibrated,lambda:model.logits(q,'text'),scope_receipt=scope_receipt,admission_receipt=admission,max_correction=1.)[0]
 descriptors=F.normalize(teacher['descriptors'].float().cuda(),dim=1);examples=[];frames=[f for f in info['train_frames'] if f in ground['frames']];held=set(frames[::5])
 with torch.no_grad():
  for frame in frames:
   vi=info['frames'].index(frame);j=ground['frames'].index(frame);target=ground['probability'][j].flatten(1).T.cuda();eligible=(ground['scores'][j].cuda()>=.7)&((target>.5).sum(0)>=20)
   ids=torch.where(ep['proposal_views']==vi)[0];masks=ep['masks'][ids].flatten(1).cuda();similarity=descriptors[ids]@q.T
   evidence=[]
   for start in range(0,len(q),8):evidence.append(similarity[:,None,start:start+8].expand(-1,masks.shape[1],-1).masked_fill(~masks[:,:,None],-1).amax(0))
   evidence=torch.cat(evidence,1);negative=torch.zeros_like(target,dtype=torch.bool)
   for ci in torch.where(eligible)[0]:
    floor=torch.quantile(evidence[target[:,ci]>.5,ci],.25)-.03
    negative[:,ci]=masks.any(0)&(evidence[:,ci]<floor)&(target[:,ci]<.2)
   positive=(target>=.8)&eligible[None];known=positive|negative
   hit=load(root/auth['views'][vi]['relative_path']);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();w=hit['base_weights'].float().cuda();total=torch.zeros(len(target),device='cuda').scatter_add_(0,pid,w);w*=model.observed_for('text')[gid];mass=torch.zeros_like(total).scatter_add_(0,pid,w);op=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(len(target),len(model.latent))).coalesce();rendered=torch.sparse.mm(op,logits)
   valid=positive.any(1)&negative.any(1)&(mass/total.clamp_min(1e-8)>.5)&(total>1e-8)
   if not valid.any():continue
   counts=positive[valid].sum(0).clamp_min(1).float();weight=(positive[valid]/counts.sqrt()).sum(1)/positive[valid].sum(1).clamp_min(1)
   examples.append((rendered[valid].cpu(),positive[valid].cpu(),known[valid].cpu(),weight.cpu(),frame in held))
   print(a.key,frame,'positive and negative competition pixels',int(valid.sum()),flush=True)
 del model,logits,descriptors
 train=[r for r in examples if not r[-1]];validation=[r for r in examples if r[-1]]
 def stack(rows):return [torch.cat([r[i] for r in rows]).cuda() for i in range(4)]
 x,pos,known,weight=stack(train);vx,vpos,vknown,vweight=stack(validation);direction=torch.nn.Parameter(torch.zeros(q.shape[1],device='cuda'));opt=torch.optim.Adam([direction],lr=.01)
 def loss(xx,pp,kk,ww):
  z=xx+(q@direction).tanh()[None]
  value=torch.logsumexp(z.masked_fill(~kk,-1e4),1)-torch.logsumexp(z.masked_fill(~pp,-1e4),1)
  return (value*ww).sum()/ww.sum().clamp_min(1e-8)
 initial=float(loss(vx,vpos,vknown,vweight).detach())
 for step in range(300):
  ix=torch.randint(len(x),(2048,),device='cuda');objective=loss(x[ix],pos[ix],known[ix],weight[ix])+.1*(q@direction).tanh().square().mean();opt.zero_grad();objective.backward();opt.step()
 state=dict(direction=direction.detach().cpu(),source_queries=ground['queries']);data['category_competition']=state
 out=Path('output/category_competition_next38')/a.key;out.mkdir(parents=True,exist_ok=True);torch.save(data,out/'field.pt');shutil.copyfile(source/'calibration.pt',out/'calibration.pt')
 report=dict(key=a.key,source_vocabulary_unchanged=True,benchmark_queries_in_training=False,unknown_labels_excluded=True,multiple_positive_labels_retained=True,train_pixels=len(x),validation_pixels=len(vx),validation_frames=sorted(held),validation_partial_loss_before=initial,validation_partial_loss_after=float(loss(vx,vpos,vknown,vweight).detach()),max_logit_correction=1.)
 (out/'training.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':main()
