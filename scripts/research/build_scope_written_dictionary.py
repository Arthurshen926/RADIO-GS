"""Write native semantic evidence through inverse-rendered scope memberships.

Keep the original dictionary as residual evidence. Matched region descriptors
remain separate at three spatial granularities instead of being pooled through
2D overlap before lifting. No query or benchmark annotation selects a carrier.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--local',action='store_true');a=p.parse_args();torch.set_num_threads(4)
 original=Path('output/scope_query_modes_next34')/a.key;data=load(original/'field.pt');scope_source=Path('output/local_rendered_scopes_next37' if a.local else 'output/rendered_scopes_next37')/a.key;graph=load(scope_source/'field.pt')['scope_hypergraph'];n,h=graph['shape'];old_count=len(data['codebook'])
 indices=graph['indices'].cuda();weight=graph['weights'].cuda();area=torch.zeros(h,device='cuda').scatter_add_(0,indices[1],weight)
 core=weight>=.5;row=indices[0,core];col=indices[1,core];membership=weight[core];cost=area[col].clamp_min(1).log()
 low=torch.full((n,),float('inf'),device='cuda').scatter_reduce_(0,row,cost,reduce='amin',include_self=True)
 high=torch.full((n,),-float('inf'),device='cuda').scatter_reduce_(0,row,cost,reduce='amax',include_self=True)
 selected=[]
 for distance in [cost-low[row],(cost-(low[row]+high[row])/2).abs(),high[row]-cost]:
  minimum=torch.full((n,),float('inf'),device='cuda').scatter_reduce_(0,row,distance,reduce='amin',include_self=True)
  equal=distance==minimum[row];chosen=torch.full((n,),h,device='cuda',dtype=torch.long).scatter_reduce_(0,row[equal],col[equal],reduce='amin',include_self=True)
  strengths=torch.zeros(n,device='cuda').scatter_reduce_(0,row,torch.where(col==chosen[row],membership,0),reduce='amax',include_self=True)
  selected.append((chosen.clamp_max(h-1),strengths))
 ids=torch.stack([x[0] for x in selected],1).cpu();strength=torch.stack([x[1] for x in selected],1).cpu();old_weight=data['weights'].float();old_weight/=old_weight.sum(-1,keepdim=True).clamp_min(1e-8)
 data['indices']=torch.cat([data['indices'].long(),(ids+old_count)[:,:,None]],2)
 data['weights']=torch.cat([old_weight*(1-strength[:,:,None]),strength[:,:,None]],2)
 data['codebook']=torch.cat([data['codebook'].half(),graph['prototypes'].half()])
 prior=torch.zeros(h,3)
 for layer in range(3):prior[:,layer].scatter_add_(0,ids[:,layer],strength[:,layer])
 prior/=prior.sum(1,keepdim=True).clamp_min(1e-8)
 data['scale_prior']=torch.cat([data['scale_prior'],prior]);data['native_modalities']=['text'];data['method']='inverse-scope native evidence writing with original evidence retained'
 out=Path('output/scope_written_'+('local' if a.local else 'matched')+'_next38')/a.key;out.mkdir(parents=True,exist_ok=True);torch.save(data,out/'field.pt')
 report=dict(key=a.key,scopes=h,original_words=old_count,new_words=len(data['codebook']),elements_with_scope_core=int((strength>0).any(1).sum()),original_evidence_retained=True,geometry_frozen=True,source_only=True,local=a.local)
 (out/'construction.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':main()
