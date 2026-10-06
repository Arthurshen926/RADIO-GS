"""Fixed native region teacher: alphaT, mass/quantile gate, robust fusion."""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.source_kernel_extent import attach_kernel_native
from radio_gs.querying.visibility_semantic_writing import visibility_gate,geometric_median,two_view_prototypes

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--teacher',choices=['region','radio_language'],default='region');p.add_argument('--dense-cache');p.add_argument('--output');p.add_argument('--two-prototypes',action='store_true');a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
    out=Path(a.output or ('output/clean_writing_next46' if a.teacher=='region' else 'output/dense_language_writing_next46'));data=load(Path('output/odds_kernel_extent_next40')/a.key/'field.pt');data.pop('source_kernel_extent');root=Path(data['root'])
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
    ep=load(root/'episodes.pt');teacher=load(root/info['teacher_files']['siglip']['path']);masks=ep['masks'];views=ep['proposal_views'];native=F.normalize(teacher['descriptors'].float().cuda(),dim=-1)
    parent=load(data['base_field']);observed=parent['state_dict']['observed'].cuda();n=len(observed);d=native.shape[1]
    if a.teacher=='radio_language' and not a.dense_cache:
        from radio_gs.models.siglip_projection import SigLIP2SummaryHead
        head=SigLIP2SummaryHead.from_radio_checkpoint('/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar').cuda().half().eval().requires_grad_(False)
        radio=load(root/'teacher_radio.pt')
        (out/a.key).mkdir(parents=True,exist_ok=True)
    sums={k:torch.zeros(n,d) for k in ['alpha','gate']};masses={k:torch.zeros(n,device='cuda') for k in sums};records=[];stats=[]
    with torch.inference_mode():
        for frame in info['train_frames']:
            vi=info['frames'].index(frame);ids=torch.where(views==vi)[0];mask=masks[ids].flatten(1).float().cuda()
            # Same region teacher for every arm; equal mask overlap mixture.
            count=mask.sum(0);grid=F.normalize(mask.T@native[ids]/count[:,None].clamp_min(1),dim=-1)
            if a.teacher=='radio_language':
                if a.dense_cache:
                    record=load(Path(a.dense_cache)/a.key/f'dense_{frame}.pt');rh,rw=record['shape'];token_features=record['features'].float().cuda()
                    (out/a.key).mkdir(parents=True,exist_ok=True)
                else:
                    raw_tokens=radio[vi].cuda();rh,rw=raw_tokens.shape[-2:]
                    # Experimental language head on spatial tokens.
                    token_features=F.normalize(head(raw_tokens.flatten(1).T[None]).float(),dim=-1)
                torch.save(dict(features=token_features.half().cpu(),shape=[rh,rw]),out/a.key/f'dense_{frame}.pt')
                grid=F.normalize(F.interpolate(token_features.transpose(1,2).reshape(1,d,rh,rw),size=(h,w),mode='bilinear',align_corners=False)[0].flatten(1).T,dim=-1)
                count=torch.ones(h*w,device='cuda')
            hit=load(root/auth['views'][vi]['relative_path']);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();raw=hit['base_weights'].float().cuda()
            keep=visibility_gate(pid,raw,h*w);weight=raw*observed[gid]*(count[pid]>0)
            for mode in sums:
                ww=weight if mode=='alpha' else weight*keep
                unique,inverse=torch.unique(gid,return_inverse=True)
                op=torch.sparse_coo_tensor(torch.stack([inverse,pid]),ww,(len(unique),h*w)).coalesce();per=torch.sparse.mm(op,grid);mass=torch.zeros(len(unique),device='cuda').scatter_add_(0,inverse,ww)
                sums[mode].index_add_(0,unique.cpu(),per.cpu());masses[mode].index_add_(0,unique,mass)
                if mode=='gate':
                    valid=mass>0;records.append((unique[valid].cpu(),F.normalize(per[valid],dim=-1).half().cpu(),mass[valid].cpu()))
            stats.append(dict(frame=frame,hit_fraction=float(keep.float().mean()),retained_mass=float((raw*keep).sum()/raw.sum())))
            print(a.key,frame,stats[-1],flush=True)
        features={k:F.normalize(v,dim=-1) for k,v in sums.items()};del sums
        features['robust']=torch.empty_like(features['gate'])
        if a.two_prototypes:features['multimodal']=torch.empty(n,2,d)
        for start_row in range(0,n,8192):
            stop=min(n,start_row+8192);chunk=[]
            for ids,feat,mass in records:
                chosen=(ids>=start_row)&(ids<stop)
                chunk.append(((ids[chosen]-start_row).cuda(),feat[chosen].float().cuda(),mass[chosen].cuda()))
            features['robust'][start_row:stop]=geometric_median(chunk,features['gate'][start_row:stop].cuda()).cpu()
            if a.two_prototypes:features['multimodal'][start_row:stop]=two_view_prototypes(chunk,features['gate'][start_row:stop].cuda()).cpu()
        masses['robust']=masses['gate'];del records
        if a.two_prototypes:masses['multimodal']=masses['gate']
    cache=load(Path('output/odds_kernel_extent_next40')/a.key/'source_training.pt');q=F.normalize(cache['query'].cuda(),dim=-1);fallback=cache['base_logits'].cuda().sigmoid()
    model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.eval().requires_grad_(False);attach_kernel_native(model,data)
    with torch.no_grad():threshold=model.native_query_thresholds['text'](q)
    examples=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in e.items()} for e in cache['examples'] if e['view'] not in cache['held']]
    for mode,feat in features.items():
        supported=(masses[mode]>0)&observed
        with torch.no_grad():cosine=torch.cat([part.cuda()@q.T for part in feat.split(8192)])
        if cosine.ndim==3:cosine=cosine.amax(1)
        scale=torch.nn.Parameter(torch.tensor(3.,device='cuda'));offset=torch.nn.Parameter(torch.tensor(0.,device='cuda'));opt=torch.optim.Adam([scale,offset],lr=.02)
        for step in range(100):
            e=examples[step%len(examples)];col=e['columns'];value=(scale.exp()*(cosine[:,col]-threshold[col]+offset)).sigmoid();value=torch.where(supported[:,None],value,fallback[:,col]);pred=torch.sparse.mm(e['op'],value);y=e['target'];pos=e['known']*y;neg=e['known']*(1-y)
            loss=(((pred-y).square()*pos).sum()/pos.sum().clamp_min(1)+((pred-y).square()*neg).sum()/neg.sum().clamp_min(1))/2
            opt.zero_grad();loss.backward();opt.step()
        state=dict(features=feat.half().cpu(),gaussian_to_element=torch.arange(n),supported=supported.cpu(),scale=float(scale.exp()),offset=float(offset))
        dest=out/mode/a.key;dest.mkdir(parents=True,exist_ok=True);torch.save(dict(data,dense_surface_field=state),dest/'field.pt')
        (dest/'construction.json').write_text(json.dumps(dict(key=a.key,mode=mode,teacher=a.teacher,region_policy='equal overlap pixel mixture' if a.teacher=='region' else None,dense_cache=a.dense_cache,dense_policy='experimental summary MLP on raw patch tokens; no SAM pooling' if a.teacher=='radio_language' else None,geometry_changed=False,query_independent_features=True,source_only_calibration=True,source_frames=info['train_frames'],gate_mass=.7,gate_retained_fraction=.5,gate_floor=1e-4,robust='5 normalized Weiszfeld steps on chord distance',support=int(supported.sum()),views=stats,seconds=time.monotonic()-start),indent=2)+'\n')
        print(a.key,mode,'saved',flush=True)

if __name__=='__main__':main()
