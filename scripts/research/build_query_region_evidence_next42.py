"""Query-free, source-selected C/F/B bank; one frozen encoder load for 3 pilots."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.querying.query_region_evidence import load,digest,same_window,normalize_membership,held_phrase
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime

KEYS=['lerf_figurines','lerf_ramen','scannet_scene0000_00']

def tensor_hash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()

def spatial_support(key,p,root,auth,views,ms,v):
    n=len(p['observed']);pixels=ms.shape[-1]*ms.shape[-2];row=[];col=[];values=[];visibility=torch.zeros(n);quality=torch.zeros(len(ms))
    for vi in views:
        hit=load(root/auth['views'][vi]['relative_path']);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();w=hit['base_weights'].float().cuda()*p['observed'][hit['gaussian_ids'].long()].cuda()
        mass=torch.zeros(pixels,device='cuda').scatter_add_(0,pid,w);op=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce();adj=op.T.coalesce();den=torch.sparse.sum(op,dim=0).to_dense();visibility+=(den>1e-8).cpu()
        ids=torch.where(v==vi)[0]
        for part in ids.split(16):
            target=ms[part].flatten(1).float().T.cuda();value=torch.sparse.mm(adj,target)/den[:,None].clamp_min(1e-8)
            for _ in range(20):value.add_(.95*torch.sparse.mm(adj,target-torch.sparse.mm(op,value))/den[:,None].clamp_min(1e-8)).clamp_(0,1)
            predicted=torch.sparse.mm(op,value);quality[part]=(1-((predicted-target).square().sum(0)/target.sum(0).clamp_min(1))).clamp(0,1).cpu()
            rr,cc=torch.where(value>1e-4);row.append(rr.cpu());col.append(part[cc.cpu()]);values.append(value[rr,cc].cpu())
        print(key,'support view',vi,flush=True)
    indices=torch.stack([torch.cat(row),torch.cat(col)]);weights=normalize_membership(indices,torch.cat(values),v,n)/visibility[indices[0]].clamp_min(1)
    flat=ms.flatten(1).float();disjoint=(flat@flat.T==0)&(v[:,None]==v[None,:]);disjoint.fill_diagonal_(False)
    return indices,weights,quality,disjoint


def build(key,out,runtime=None):
    start=time.monotonic();directory=out/key;directory.mkdir(parents=True,exist_ok=True)
    ppath=Path('output/taskwise_scope_composition_next38')/key/'field.pt';p=load(ppath);root=Path(p['root'])
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());src=json.loads((root/'source_frames.json').read_text())
    views=[info['frames'].index(f) for f in info['train_frames']];held=views[::5]
    ep=load(root/'episodes.pt');cap=load(root/'caption_region_evidence.pt');teacher=load(root/info['teacher_files']['siglip']['path'])
    masks=[ep['masks'].bool(),cap['masks'].bool()];vs=[ep['proposal_views'],cap['proposal_views']]
    tight=[teacher['descriptors'].float(),cap['native_descriptors'].float()]
    kinds=['sam']*len(masks[0])+['caption']*len(masks[1]);localids=list(range(len(masks[0])))+list(range(len(masks[1])))
    offset=len(masks[0]);small=Path('output/scope_hierarchy_full_next33')/key/'small_teacher.pt'
    inputs=[ppath,root/'episodes.pt',root/'caption_region_evidence.pt',root/'teachers.json',root/'responsibility.json',root/'source_frames.json',root/info['teacher_files']['siglip']['path'],Path(p['base_field']),Path(p['geometry_report']),ppath.parent/'calibration.pt']
    if small.exists():
        extra=load(small);masks.append(extra['masks'].bool());vs.append(extra['proposal_views']);tight.append(extra['descriptors'].float());kinds+=['small']*len(extra['masks']);localids+=list(range(len(extra['masks'])));inputs.append(small)
    masks=torch.cat(masks);vs=torch.cat(vs).long();tight=torch.cat(tight);area=masks.flatten(1).sum(1)
    # View-balanced round-robin over source kind and area quartile. No phrase read.
    buckets=[]
    for vi in views:
        ids=[i for i in torch.where((vs==vi)&(area>0))[0].tolist() if kinds[i]!='caption' or cap['records'][localids[i]]['confidence']>=.7]
        ids=sorted(ids,key=lambda i:(int(area[i]),kinds[i],localids[i]))
        groups={}
        for rank,i in enumerate(ids):groups.setdefault((kinds[i],min(3,4*rank//max(len(ids),1))),[]).append(i)
        ordered=[]
        while any(groups.values()):
            for group in sorted(groups):
                if groups[group]:ordered.append(groups[group].pop(0))
        buckets.append(ordered)
    selected=[]
    while len(selected)<512 and any(buckets):
        for bucket in buckets:
            if bucket and len(selected)<512:selected.append(bucket.pop(0))
    selected=torch.tensor(selected);ms=masks[selected];v=vs[selected];tight=tight[selected]
    # Only native vector evidence is deployed; labels remain training provenance.
    text=torch.zeros(len(selected),cap['text_embeddings'].shape[1]);associations=[]
    for j,i in enumerate(selected.tolist()):
        if kinds[i]=='caption':qs=cap['records'][localids[i]]['phrase_indices']
        else:
            ids=torch.where(cap['proposal_views']==vs[i])[0];cm=cap['masks'][ids].bool();inter=(cm & masks[i]).flatten(1).sum(1);union=(cm|masks[i]).flatten(1).sum(1)
            matched=ids[inter/union.clamp_min(1)>.5];qs=sorted({q for k in matched.tolist() for q in cap['records'][k]['phrase_indices']})
        associations.append(qs)
        if qs:text[j]=F.normalize(cap['text_embeddings'][qs].float().mean(0),dim=0)
    # Snapshot all legal authority/RGB inputs before encoding/support.
    frames={r['frame_idx']:r for r in src['frames']}
    for vi in views:
        inputs.extend([root/auth['views'][vi]['relative_path'],Path(src['image_dir'])/frames[info['frames'][vi]]['source_file']])
    bindings=[dict(path=str(x),sha256=digest(x)) for x in sorted(set(inputs))]
    (directory/'inputs_before.json').write_text(json.dumps(bindings,indent=2))
    (directory/'selection.json').write_text(json.dumps(dict(cap=512,actual=len(selected),held_views=held,held_phrases=[i for i,s in enumerate(cap['phrases']) if held_phrase(s)],selection='view balanced, source kind and area quartile round robin; no benchmark inputs',mask_resolution=list(ms.shape[-2:]),intervention='1.5x same box, bicubic 384, neutral RGB 0.5; source 96x128 masks nearest upsampled to actual RGB',kinds={k:sum(kinds[i]==k for i in selected.tolist()) for k in set(kinds)}),indent=2))
    if runtime is None:runtime=NativeSiglip2Runtime.load(Path(teacher['encoder']['path']),device=torch.device('cuda'))
    else:assert str(Path(teacher['encoder']['path']).resolve())==runtime.bundle['path']
    descriptors=torch.empty(len(selected),3,tight.shape[1]);records=[];encode_seconds=0
    for vi in views:
        frame=frames[info['frames'][vi]];path=Path(src['image_dir'])/frame['source_file'];sha=digest(path);assert sha==frame['source_sha256']
        rgb=torch.from_numpy(np.array(Image.open(path).convert('RGB'),copy=True)).permute(2,0,1).float()/255
        ids=torch.where(v==vi)[0];crops=[]
        for j in ids.tolist():
            mask=F.interpolate(ms[j][None,None].float(),size=rgb.shape[1:],mode='nearest-exact')[0,0].bool()
            crop,box=same_window(rgb,mask);crops.append(crop)
            records.append(dict(bank_index=j,origin=kinds[int(selected[j])],origin_index=localids[int(selected[j])],source_view=vi,frame=info['frames'][vi],rgb_path=str(path),rgb_sha256=sha,mask_sha256=tensor_hash(mask),stored_mask_sha256=tensor_hash(ms[j]),box=box,crop_sha256=tensor_hash(crop),encoder_sha256=runtime.bundle['sha256']))
        batch=torch.cat(crops);t=time.monotonic()
        with torch.inference_mode():encoded=torch.cat([runtime.encode(c).cpu() for c in batch.split(8)])
        torch.cuda.synchronize();encode_seconds+=time.monotonic()-t;descriptors[ids]=encoded.reshape(len(ids),3,-1)
        print(key,'encoded view',vi,'regions',len(ids),flush=True)
    indices,weights,quality,disjoint=spatial_support(key,p,root,auth,views,ms,v)
    bank=dict(schema='query_region_evidence.next42.v1',indices=indices.int(),weights=weights,shape=[len(p['observed']),len(selected)],views=v,held_views=held,C=descriptors[:,0].half(),F=descriptors[:,1].half(),B=descriptors[:,2].half(),tight=tight.half(),text=text.half(),quality=quality,area=ms.flatten(1).float().mean(1),disjoint=disjoint,masks=ms,associations=associations,records=records,encoder=runtime.bundle,geometry_binding=p['geometry_binding'],source_only=True)
    torch.save(bank,directory/'bank.pt')
    after=[dict(path=x['path'],sha256=digest(x['path'])) for x in bindings];assert after==bindings
    (directory/'inputs_after.json').write_text(json.dumps(after,indent=2))
    (directory/'build.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,encoding_seconds=encode_seconds,bank_bytes=(directory/'bank.pt').stat().st_size,regions=len(selected),nnz=len(weights),encoder=runtime.bundle,inputs_unchanged=True,source_only=True,benchmark_data_used=False),indent=2))
    return runtime

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);args=a.parse_args();torch.set_num_threads(4)
    runtime=None
    for key in KEYS:runtime=build(key,Path(args.output),runtime)
