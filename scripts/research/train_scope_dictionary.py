"""Fit sparse native code weights through the source pixel rendering operator."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.native_scope_dictionary import pixel_scope_owners


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--base',default='output/scope_dictionary_train_next33');p.add_argument('--steps',type=int,default=400);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    folder=Path(a.base)/a.key;payload=load(folder/'field.pt');root=Path(payload['root'])
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    teacher=load(root/info['teacher_files']['siglip']['path']);ep=load(root/'episodes.pt')
    descriptors=F.normalize(teacher['descriptors'].float(),dim=1);masks=ep['masks'];views=ep['proposal_views']
    extra=Path('output/small_region_restoration_next32')/a.key/'small_teacher.pt'
    if (folder/'small_teacher.pt').exists():extra=folder/'small_teacher.pt'
    if extra.exists():
        small=load(extra);descriptors=torch.cat([descriptors,F.normalize(small['descriptors'].float(),dim=1)])
        masks=torch.cat([masks,small['masks']]);views=torch.cat([views,small['proposal_views']])
    levels=load(folder/'training.pt')['levels'];native=descriptors.cuda()
    codebook=F.normalize(payload['codebook'].cuda().float(),dim=1);gram=codebook@codebook.T
    indices=payload['indices'].long().cuda();initial=payload['weights'].float().cuda()
    allowed=initial>0;initial=initial/initial.sum(-1,keepdim=True).clamp_min(1e-8)
    logits=torch.nn.Parameter(initial.clamp_min(1e-8).log());optimizer=torch.optim.Adam([logits],lr=.04)
    observed=payload['observed'].cuda();n=len(observed)
    text=F.normalize(teacher['text_embeddings'].float().cuda(),dim=1)
    trainmask=torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in views])
    image=native[trainmask]
    source=[]
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);hit=load(root/auth['views'][vi]['relative_path'])
        pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();w=hit['base_weights'].float().cuda()
        unique,inverse=torch.unique(gid,return_inverse=True)
        pixel_count=masks.shape[-2]*masks.shape[-1]
        mass=torch.zeros(pixel_count,device='cuda').scatter_add_(0,pid,w*observed[gid])
        operator=torch.sparse_coo_tensor(torch.stack([pid,inverse]),w*observed[gid]/mass[pid].clamp_min(1e-8),(pixel_count,len(unique))).coalesce()
        view_ids=torch.where(views==vi)[0]
        if payload.get('granularity')=='hierarchy':owners,covered=pixel_scope_owners(masks[view_ids].flatten(1).cuda())
        for level in range(3):
            if payload.get('granularity')=='hierarchy':
                target=native[view_ids.cuda()[owners[level]]]
                valid=covered&(mass>1e-8)
                source.append(dict(rows=unique.cpu(),operator=operator.cpu(),target=target.half().cpu(),valid=valid.cpu(),level=level))
                continue
            ids=torch.where((views==vi)&(levels==level))[0]
            if not len(ids):continue
            mask=masks[ids].flatten(1).float().cuda();partition=mask/mask.sum(0).clamp_min(1)[None]
            target=F.normalize(partition.T@native[ids],dim=1)
            # The teacher is defined only where this scale has source evidence.
            valid=(mask.sum(0)>0)&(mass>1e-8)
            source.append(dict(rows=unique.cpu(),operator=operator.cpu(),target=target.half().cpu(),valid=valid.cpu(),level=level))
    history=[]
    for step in range(a.steps):
        sample=source[step%len(source)];rows=sample['rows'].cuda();level=sample['level'];operator=sample['operator'].cuda()
        ids=indices[rows,level];permit=allowed[rows,level];weights=logits[rows,level].masked_fill(~permit,-30).softmax(1)
        has=permit.any(1);weights=weights*has[:,None]
        query=torch.cat([text[torch.randint(len(text),(24,),device='cuda')],image[torch.randint(len(image),(24,),device='cuda')]])
        values=(codebook@query.T)[ids]
        norm=(gram[ids[:,:,None],ids[:,None,:]]*weights[:,:,None]*weights[:,None,:]).sum((1,2)).clamp_min(1e-8).sqrt()
        scores=(values*weights[...,None]).sum(1)/norm[:,None]
        rendered=torch.sparse.mm(operator,scores)
        coverage=torch.sparse.mm(operator,has.float()[:,None]).flatten()
        valid=sample['valid'].cuda()&(coverage>.5)
        rendered=rendered/coverage[:,None].clamp_min(1e-8)
        target=sample['target'].cuda().float()@query.T
        reconstruction=(rendered[valid]-target[valid]).square().mean()
        prior=initial[rows,level]
        regularizer=(weights*(weights.clamp_min(1e-8).log()-prior.clamp_min(1e-8).log())).sum(1)[has].mean()
        loss=100*reconstruction+.001*regularizer
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%50==0:
            row=dict(step=step+1,response_mse=float(reconstruction),kl=float(regularizer));history.append(row);print(a.key,row,flush=True)
    with torch.no_grad():
        weights=logits.masked_fill(~allowed,-30).softmax(-1)*allowed.any(-1)[...,None]
    payload['weights']=weights.cpu().half();payload['training_steps']=a.steps
    out=folder/'trained';out.mkdir(exist_ok=True)
    torch.save(payload,out/'field.pt')
    (out/'training.json').write_text(json.dumps(dict(key=a.key,steps=a.steps,source_views=len(info['train_frames']),native_codebook_frozen=True,geometry_frozen=True,history=history),indent=2)+'\n')


if __name__=='__main__':main()
