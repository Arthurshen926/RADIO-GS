"""Boost source scopes with reciprocal overlap in jointly visible geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from radio_gs.models.native_scope_dictionary import pixel_scope_owners


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--source',required=True);p.add_argument('--output',default='output/scope_geometric_next34');a=p.parse_args()
    torch.set_num_threads(4);source=Path(a.source)/a.key;data=load(source/'field.pt');root=Path(data['root'])
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());ep=load(root/'episodes.pt')
    teacher=load(root/info['teacher_files']['siglip']['path']);native=teacher['descriptors'].float();masks=ep['masks'];views=ep['proposal_views']
    extra=source/'small_teacher.pt'
    if not extra.exists():extra=Path('output/small_region_restoration_next32')/a.key/'small_teacher.pt'
    small=load(extra);native=torch.cat([native,small['descriptors'].float()]);masks=torch.cat([masks,small['masks']]);views=torch.cat([views,small['proposal_views']])
    n=len(data['observed']);r=len(masks);v=len(info['frames']);training=[info['frames'].index(f) for f in info['train_frames']]
    rows=[];cols=[];visible_rows=[];visible_cols=[]
    def operator(vi):
        hit=load(root/auth['views'][vi]['relative_path']);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();w=hit['base_weights'].float().cuda()
        mass=torch.zeros(masks.shape[-2]*masks.shape[-1],device='cuda').scatter_add_(0,pid,w)
        return torch.sparse_coo_tensor(torch.stack([gid,pid]),w/mass[pid].clamp_min(1e-8),(n,len(mass))).coalesce()
    for vi in training:
        ids=torch.where(views==vi)[0];op=operator(vi);mass=torch.sparse.sum(op,dim=1).to_dense();visible=mass>=.25
        membership=torch.sparse.mm(op,masks[ids].flatten(1).float().cuda().T)/mass[:,None].clamp_min(1e-8)
        gi,ri=torch.where((membership>.5)&visible[:,None]);rows.append(ids[ri.cpu()].numpy());cols.append(gi.cpu().numpy())
        gi=torch.where(visible)[0].cpu().numpy();visible_rows.append(np.full(len(gi),vi));visible_cols.append(gi)
        print(a.key,'geometric footprints',info['frames'][vi],flush=True)
    del op,mass,membership,visible
    rr=np.concatenate(rows);cc=np.concatenate(cols);x=sp.csr_matrix((np.ones(len(rr),np.float32),(rr,cc)),shape=(r,n))
    rr=np.concatenate(visible_rows);cc=np.concatenate(visible_cols);visibility=sp.csr_matrix((np.ones(len(rr),np.float32),(rr,cc)),shape=(v,n))
    intersection=(x@x.T).toarray();shared=(x@visibility.T).toarray();area=np.asarray(x.sum(1)).ravel()
    view_ids=views.numpy();a_visible=shared[:,view_ids];b_visible=a_visible.T
    iou=intersection/np.maximum(a_visible+b_visible-intersection,1)
    compatible=(intersection>=.7*a_visible)&(intersection>=.7*b_visible)&(a_visible>=.25*area[:,None])&(b_visible>=.25*area[None])&(intersection>0)&(view_ids[:,None]!=view_ids[None])
    quality=np.where(compatible,iou,0)
    best=np.full((r,v),-1,np.int64)
    for vi in training:
        ids=np.where(view_ids==vi)[0];best[:,vi]=ids[quality[:,ids].argmax(1)]
    evidence=np.zeros(r,np.float32);pairs=0
    for vi in training:
        other=best[:,vi];back=best[other,view_ids];score=quality[np.arange(r),other]
        reciprocal=(back==np.arange(r))&(score>=.5);evidence+=score*reciprocal;pairs+=int(reciprocal.sum())
    available=((shared>=.25*area[:,None])&(shared>0)).sum(1)-1
    gain=(1+evidence/np.maximum(available,1)).astype(np.float32)
    # Lack of a match is unknown: retain the original observation at weight one.
    book=F.normalize(data['codebook'].cuda().float(),dim=1);codes=(F.normalize(native.cuda(),dim=1)@book.T).argmax(1);gain_cuda=torch.from_numpy(gain).cuda()
    hist=torch.zeros(3,n,len(book),device='cuda')
    for vi in training:
        ids=torch.where(views==vi)[0].cuda();owners,covered=pixel_scope_owners(masks[ids.cpu()].flatten(1).cuda());op=operator(vi)
        for level in range(3):
            region=ids[owners[level]];color=F.one_hot(codes[region],len(book)).float()*(covered*gain_cuda[region])[:,None]
            hist[level]+=torch.sparse.mm(op,color)
    values,indices=hist.topk(4,dim=2);data.update(indices=indices.permute(1,0,2).cpu().short(),weights=(values/values.sum(2,keepdim=True).clamp_min(1e-8)).permute(1,0,2).cpu().half(),fusion='reciprocal_joint_visibility_scope_support')
    out=Path(a.output)/a.key;out.mkdir(parents=True,exist_ok=True);torch.save(data,out/'field.pt')
    report=dict(key=a.key,reciprocal_pairs=pairs//2,regions_with_support=int((evidence>0).sum()),training_regions=int(sum(np.isin(view_ids,training))),mean_supported_gain=float(gain[evidence>0].mean()) if (evidence>0).any() else 1.,geometry_frozen=True,semantic_similarity_used_for_correspondence=False)
    (out/'correspondence.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':main()
