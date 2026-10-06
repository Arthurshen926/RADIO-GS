"""Build distinct cross-view scope hypotheses, validated in reserved source views."""
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--folds',type=int,default=1);p.add_argument('--fixed-cores',action='store_true');p.add_argument('--compatible-scopes',action='store_true');p.add_argument('--output',default='output/scope_hypergraph_next36');a=p.parse_args();torch.set_num_threads(4)
 source=Path('output/nvos_scope_init_next35' if a.key.startswith('nvos_') else 'output/scope_query_modes_next34')/a.key
 data=load(source/'field.pt');root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());ep=load(root/'episodes.pt');teacher=load(root/info['teacher_files']['siglip']['path'])
 masks=ep['masks'];views=ep['proposal_views'];native=teacher['descriptors'].float()
 extra=Path('output/scope_hierarchy_full_next33')/a.key/'small_teacher.pt'
 if not extra.exists():extra=Path('output/small_region_restoration_next32')/a.key/'small_teacher.pt'
 if extra.exists():
  small=load(extra);masks=torch.cat([masks,small['masks']]);views=torch.cat([views,small['proposal_views']]);native=torch.cat([native,small['descriptors'].float()])
 n=len(data['observed']);r=len(masks);v=len(info['frames']);train=[info['frames'].index(f) for f in info['train_frames']];held=[info['frames'].index(f) for f in info['validation_frames']]
 # Reserve views before any correspondence or footprint fusion.
 if not held:held=train[-2:];train=train[:-2]
 row=[];col=[];vr=[];vc=[]
 for vi in train+held:
  hit=load(root/auth['views'][vi]['relative_path']);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();w=hit['base_weights'].float().cuda();mass=torch.zeros(masks.shape[-2]*masks.shape[-1],device='cuda').scatter_add_(0,pid,w)
  op=torch.sparse_coo_tensor(torch.stack([gid,pid]),w/mass[pid].clamp_min(1e-8),(n,len(mass))).coalesce();gm=torch.sparse.sum(op,dim=1).to_dense();visible=gm>=.25;ids=torch.where(views==vi)[0]
  membership=torch.sparse.mm(op,masks[ids].flatten(1).float().cuda().T)/gm[:,None].clamp_min(1e-8)
  gi,ri=torch.where((membership>.5)&visible[:,None]);row.append(ids[ri.cpu()].numpy());col.append(gi.cpu().numpy());gi=torch.where(visible)[0].cpu().numpy();vr.append(np.full(len(gi),vi));vc.append(gi)
  print(a.key,'scope footprints',vi,flush=True)
 x=sp.csr_matrix((np.ones(sum(map(len,row)),np.float32),(np.concatenate(row),np.concatenate(col))),shape=(r,n));visibility=sp.csr_matrix((np.ones(sum(map(len,vr)),np.float32),(np.concatenate(vr),np.concatenate(vc))),shape=(v,n))
 def hypotheses(train,held):
  tids=np.where(np.isin(views.numpy(),train))[0];tx=x[tids];tv=views.numpy()[tids];area=np.asarray(tx.sum(1)).ravel();shared=(tx@visibility.T).toarray();inter=(tx@tx.T).toarray();av=shared[:,tv];bv=av.T
  common=(av>=.25*area[:,None])&(bv>=.25*area[None])&(av>0)&(bv>0)&(tv[:,None]!=tv[None])
  compatible=common&(inter>=.7*av)&(inter>=.7*bv)
  quality=np.where(compatible,inter/np.maximum(av+bv-inter,1),0)
  best=np.full((len(tids),v),-1,np.int64)
  for vi in train:
   ids=np.where(tv==vi)[0]
   if len(ids):best[:,vi]=ids[quality[:,ids].argmax(1)]
  edges=[]
  for i in range(len(tids)):
   for vi in train:
    j=best[i,vi]
    if j>i and best[j,tv[i]]==i and quality[i,j]>=.5:edges.append((float(quality[i,j]),i,j))
  # A high-score bridge cannot transitively merge known incompatible scopes or
  # two proposals from the same view. Inclusion is not an identity equivalence.
  parent=np.arange(len(tids));groups={i:[i] for i in range(len(tids))};rejected=0
  def find(i):
   while parent[i]!=i:i=int(parent[i])
   return i
  for score,i,j in sorted(edges,reverse=True):
   i,j=find(i),find(j)
   if i==j:continue
   ai,aj=groups[i],groups[j]
   if set(tv[ai])&set(tv[aj]) or np.any(common[np.ix_(ai,aj)]&~compatible[np.ix_(ai,aj)]):rejected+=1;continue
   parent[j]=i;groups[i]=ai+aj;del groups[j]
  tracks=[g for g in groups.values() if len(g)>=2];rows=[];cols=[];values=[];prototypes=[];qualities=[];records=[]
  for group in tracks:
   ids=tids[group];sv=tv[group];positive=np.asarray(x[ids].sum(0)).ravel();support=np.where(positive>0)[0];den=np.asarray(visibility[sv][:,support].sum(0)).ravel();membership=positive[support]/np.maximum(den,1)
   # No geometry/membership update uses these held-out views.
   overlap=[]
   for vi in held:
    visible=visibility[vi,support].toarray().ravel()>0;footprint=support[visible&(membership>=.5)]
    if len(footprint)<8:continue
    candidates=np.where(views.numpy()==vi)[0];intersection=np.asarray(x[candidates][:,footprint].sum(1)).ravel();candidate_area=np.asarray(x[candidates].sum(1)).ravel();iou=intersection/np.maximum(len(footprint)+candidate_area-intersection,1)
    overlap.append(float(iou.max()) if len(iou) else 0.)
   validation=max(overlap,default=0.)
   if validation<.5:continue
   keep=membership>=.5;support=support[keep];membership=membership[keep];index=len(prototypes)
   rows.append(support);cols.append(np.full(len(support),index));values.append(membership.astype(np.float32));prototypes.append(F.normalize(native[ids],dim=1).mean(0));qualities.append(validation)
   records.append(dict(regions=ids.tolist(),views=sv.tolist(),validation_views=held,validation_iou=validation,elements=len(support)))
  return rows,cols,values,prototypes,qualities,records,len(edges),rejected,len(tracks)
 rows=[];cols=[];values=[];prototypes=[];qualities=[];records=[];edge_count=0;rejected_count=0;track_count=0
 all_views=sorted(train+held)
 for fold in range(a.folds):
  fit=train if a.folds==1 else [vi for j,vi in enumerate(all_views) if j%a.folds!=fold]
  validation=held if a.folds==1 else [vi for j,vi in enumerate(all_views) if j%a.folds==fold]
  rr,cc,vv,pp,qq,records_fold,ne,nr,nt=hypotheses(fit,validation)
  offset=len(prototypes);rows.extend(rr);cols.extend([c+offset for c in cc]);values.extend(vv);prototypes.extend(pp);qualities.extend(qq);records.extend([dict(record,fold=fold) for record in records_fold]);edge_count+=ne;rejected_count+=nr;track_count+=nt
 indices=(torch.from_numpy(np.stack([np.concatenate(rows),np.concatenate(cols)])) if rows else torch.empty(2,0,dtype=torch.long));weights=(torch.from_numpy(np.concatenate(values)) if values else torch.empty(0))
 descriptors=F.normalize(torch.stack(prototypes),dim=1).half() if prototypes else torch.empty(0,native.shape[1],dtype=torch.float16)
 data['scope_fixed_cores']=a.fixed_cores
 data['scope_compatible']=a.compatible_scopes
 data['scope_after_prompt']=a.fixed_cores and a.key.startswith('nvos_')
 data.update(scope_hypergraph=dict(indices=indices,weights=weights,prototypes=descriptors,quality=torch.tensor(qualities),shape=[n,len(prototypes)]),method='held-source-view validated geometric scope hypergraph')
 out=Path(a.output)/a.key;out.mkdir(parents=True,exist_ok=True);torch.save(data,out/'field.pt');report=dict(key=a.key,restored_small_regions=len(masks)-len(ep['masks']),source_frames=[info['frames'][i] for i in all_views],training_frames=[info['frames'][i] for i in train] if a.folds==1 else None,heldout_frames=[info['frames'][i] for i in held] if a.folds==1 else None,folds=a.folds,reciprocal_edges=edge_count,rejected_merges=rejected_count,multiview_tracks=track_count,validated_tracks=len(records),records=records,geometry_frozen=True)
 (out/'construction.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='records'},flush=True)


if __name__=='__main__':main()
