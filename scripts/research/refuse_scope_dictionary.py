"""Fuse source-view votes with leave-one-view-out native semantic support."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.native_scope_dictionary import pixel_scope_owners, containment_scope_owners


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--key',required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--output',default='output/scope_consensus_next34')
    p.add_argument('--scope',choices=['area','containment'],default='area')
    p.add_argument('--preserve-view-mass',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    source=Path(a.source)/a.key;payload=load(source/'field.pt')
    root=Path(payload['root']);info=json.loads((root/'teachers.json').read_text())
    auth=json.loads((root/'responsibility.json').read_text());ep=load(root/'episodes.pt')
    teacher=load(root/info['teacher_files']['siglip']['path'])
    native=teacher['descriptors'].float();masks=ep['masks'];views=ep['proposal_views']
    small_path=source/'small_teacher.pt'
    if not small_path.exists():
        small_path=Path('output/small_region_restoration_next32')/a.key/'small_teacher.pt'
    small=load(small_path)
    native=torch.cat([native,small['descriptors'].float()])
    masks=torch.cat([masks,small['masks']]);views=torch.cat([views,small['proposal_views']])
    book=F.normalize(payload['codebook'].cuda().float(),dim=1)
    codes=(F.normalize(native.cuda(),dim=1)@book.T).argmax(1)
    n=len(payload['observed']);k=len(book)
    # A view contributes at most one unit of evidence per element and level.
    # Weak footprints contribute less; repeated pixels do not become independent views.
    total=torch.zeros(3,n,k,device='cuda');count=torch.zeros(3,n,device='cuda');records=[]
    scale_counts=torch.zeros(k,3,device='cuda')
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);region_ids=torch.where(views==vi)[0].cuda()
        owner_fn=containment_scope_owners if a.scope=='containment' else pixel_scope_owners
        owners,covered=owner_fn(masks[region_ids.cpu()].flatten(1).cuda())
        hit=load(root/auth['views'][vi]['relative_path'])
        pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();w=hit['base_weights'].float().cuda()
        mass=torch.zeros(len(covered),device='cuda').scatter_add_(0,pid,w)
        op=torch.sparse_coo_tensor(torch.stack([gid,pid]),w/mass[pid].clamp_min(1e-8),(n,len(covered))).coalesce()
        levels=[]
        for level in range(3):
            pixel_codes=codes[region_ids[owners[level]]]
            color=F.one_hot(pixel_codes,k).float()*covered[:,None]
            scale_counts[:,level].scatter_add_(0,pixel_codes,covered.float())
            hist=torch.sparse.mm(op,color);amount=hist.sum(1)
            values,ids=hist.topk(8,dim=1)
            confidence=amount/(amount+1)
            votes=values/values.sum(1,keepdim=True).clamp_min(1e-8)*confidence[:,None]
            total[level].scatter_add_(1,ids,votes);count[level]+=confidence
            levels.append((ids.cpu().short(),votes.cpu(),confidence.cpu(),values.cpu()))
        records.append(levels)
        print(a.key,'view votes',frame,flush=True)
    kernel=((book@book.T-1)*20).exp()
    del hist,color,op,native,masks,ep,teacher
    torch.cuda.empty_cache()
    coherent_values=[];coherent_indices=[]
    agreements=[]
    for level in range(3):
        support=total[level]@kernel
        coherent=torch.zeros_like(total[level])
        for levels in records:
            ids,votes,confidence,raw_mass=levels[level];ids=ids.cuda().long();votes=votes.cuda();confidence=confidence.cuda()
            own=(kernel[ids[:,:,None],ids[:,None,:]]*votes[:,None,:]).sum(2)
            other=(support.gather(1,ids)-own).clamp_min(0)
            other_mass=(count[level]-confidence).clamp_min(0)
            # No independent observation means unknown, not contradiction.
            reliability=(1+other)/(1+other_mass[:,None])
            write_mass=raw_mass.cuda() if a.preserve_view_mass else votes
            coherent.scatter_add_(1,ids,write_mass*reliability)
            valid=votes>0
            agreements.append(float(reliability[valid].mean()))
        values,ids=coherent.topk(4,dim=1)
        coherent_values.append(values.cpu());coherent_indices.append(ids.cpu())
        del coherent,support
    out=Path(a.output)/a.key;out.mkdir(parents=True,exist_ok=True)
    for name in ['balanced','consensus']:
        if name=='balanced':values,ids=total.topk(4,dim=2)
        else:values,ids=torch.stack(coherent_values),torch.stack(coherent_indices)
        result=dict(payload,indices=ids.permute(1,0,2).cpu().short(),weights=(values/values.sum(2,keepdim=True).clamp_min(1e-8)).permute(1,0,2).cpu().half(),fusion=name,preserve_view_mass=a.preserve_view_mass,scope_assignment=a.scope,source_parent=str(source/'field.pt'))
        if a.scope=='containment':
            result['scale_prior']=(scale_counts/scale_counts.sum(1,keepdim=True).clamp_min(1)).cpu()
        folder=out if name=='consensus' else Path(str(a.output)+'_balanced')/a.key
        folder.mkdir(parents=True,exist_ok=True);torch.save(result,folder/'field.pt')
    (out/'fusion.json').write_text(json.dumps(dict(key=a.key,views=len(records),mean_independent_support=sum(agreements)/len(agreements),geometry_frozen=True,codebook_frozen=True,query_independent=True),indent=2)+'\n')


if __name__=='__main__':main()
