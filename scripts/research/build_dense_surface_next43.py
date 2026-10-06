"""Same dense source teacher, Gaussian and surface-carrier lifting controls."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.siglip_projection import SigLIP2FeatureProjection
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.source_kernel_extent import attach_kernel_native


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--language-head',action='store_true');p.add_argument('--native-vision',action='store_true');p.add_argument('--output',default='output/dense_surface_next43');a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(0)
    out=Path(a.output)/a.key;out.mkdir(parents=True,exist_ok=True)
    data=load(Path('output/odds_kernel_extent_next40')/a.key/'field.pt');data.pop('source_kernel_extent');root=Path(data['root'])
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
    gr=json.loads(Path(data['geometry_report']).read_text());geo=load(gr['results']['independent_scale']['geometry'])['state_dict'];n=len(geo['xyz'])
    # Separate nearby surfaces with a dominant normal-axis bin, not semantics.
    size=float(geo['scales'][:,0].median())*2
    normals=geo['normals'];axis=normals.abs().argmax(1);normal_bin=axis*2+(normals[torch.arange(n),axis]>0).long()
    keys=torch.cat([torch.floor(geo['xyz']/size).long(),normal_bin[:,None]],1)
    cells,mapping=torch.unique(keys,dim=0,return_inverse=True);mapping=mapping.cuda();count=len(cells)
    parent=load(data['base_field']);observed=parent['state_dict']['observed'].cuda()
    radio=load(root/'teacher_radio.pt')
    projection=SigLIP2FeatureProjection.from_radio_checkpoint('/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar').cuda().half().eval().requires_grad_(False)
    language_head=None
    native_vision=None
    if a.language_head:
        from transformers.models.siglip.modeling_siglip import SiglipMultiheadAttentionPoolingHead
        from transformers import SiglipVisionConfig
        from safetensors import safe_open
        bundle=Path(json.loads((root/'native_query_encoder.json').read_text())['siglip_root'])
        config=SiglipVisionConfig(**json.loads((bundle/'config.json').read_text())['vision_config'])
        language_head=SiglipMultiheadAttentionPoolingHead(config)
        index=json.loads((bundle/'model.safetensors.index.json').read_text())['weight_map'];weights={}
        for name,shard in index.items():
            if name.startswith('vision_model.head.'):
                with safe_open(bundle/shard,framework='pt') as f:weights[name.removeprefix('vision_model.head.')]=f.get_tensor(name)
        language_head.load_state_dict(weights);language_head=language_head.cuda().half().eval().requires_grad_(False)
    if a.native_vision:
        from transformers import SiglipVisionModel
        from PIL import Image
        import numpy as np
        native_vision=SiglipVisionModel.from_pretrained(str(bundle),local_files_only=True,torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
        language_head=native_vision.vision_model.head
        source=json.loads((root/'source_frames.json').read_text());rgb={r['frame_idx']:str(Path(source['image_dir'])/r['source_file']) for r in source['frames']}
    sums={mode:torch.zeros((n if mode.startswith('gaussian') else count,1536),device='cuda') for mode in ['gaussian','gaussian_gate','surface']}
    masses={mode:torch.zeros(len(v),device='cuda') for mode,v in sums.items()};source_files=[]
    with torch.inference_mode():
        for frame in info['train_frames']:
            vi=info['frames'].index(frame);x=radio[vi:vi+1].cuda();rh,rw=x.shape[-2:]
            if native_vision is None:
                feature=projection(x.flatten(2).transpose(1,2))
            else:
                image=Image.open(rgb[frame]).convert('RGB');width=512;height=max(16,round(image.height/image.width*width/16)*16)
                pixels=torch.from_numpy(np.asarray(image.resize((width,height),Image.Resampling.BICUBIC)).copy()).cuda().permute(2,0,1)[None].half()/127.5-1
                feature=native_vision(pixel_values=pixels,interpolate_pos_encoding=True).last_hidden_state;rh,rw=height//16,width//16
            if language_head is not None:
                local=F.unfold(F.pad(feature.transpose(1,2).reshape(1,1536,rh,rw),(1,1,1,1),mode='replicate'),kernel_size=3)
                local=local.reshape(1536,9,rh*rw).permute(2,1,0)
                feature=torch.cat([language_head(part) for part in local.split(128)])[None]
            feature=F.normalize(feature.float(),dim=-1)
            torch.save(dict(features=feature.half().cpu(),shape=[rh,rw]),out/f'dense_{frame}.pt')
            grid=F.interpolate(feature.transpose(1,2).reshape(1,1536,rh,rw),size=(h,w),mode='bilinear',align_corners=False)[0].flatten(1).T
            grid=F.normalize(grid,dim=-1);hit=load(root/auth['views'][vi]['relative_path'])
            pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();weight=hit['base_weights'].float().cuda()*observed[gid]
            peak=torch.zeros(h*w,device='cuda').scatter_reduce_(0,pid,weight,reduce='amax',include_self=True)
            for mode in ['gaussian','gaussian_gate']:
                ww=weight if mode=='gaussian' else weight*(weight>=.5*peak[pid])
                op=torch.sparse_coo_tensor(torch.stack([gid,pid]),ww,(n,h*w)).coalesce()
                sums[mode]+=torch.sparse.mm(op,grid);masses[mode].scatter_add_(0,gid,ww)
            # One highest-contribution recipient per source pixel; no copying
            # the same feature to every front/back overlapping Gaussian.
            owner=torch.full((h*w,),n,dtype=torch.long,device='cuda')
            selected=(weight==peak[pid])&(weight>0)
            owner.scatter_reduce_(0,pid[selected],gid[selected],reduce='amin',include_self=True)
            pix=torch.where(owner<n)[0];cid=mapping[owner[pix]];ww=peak[pix]
            op=torch.sparse_coo_tensor(torch.stack([cid,pix]),ww,(count,h*w)).coalesce()
            per=torch.sparse.mm(op,grid);mass=torch.zeros(count,device='cuda').scatter_add_(0,cid,ww)
            sums['surface']+=per;masses['surface']+=mass
            valid=mass>0;path=out/f'source_{frame}.pt';torch.save(dict(ids=torch.where(valid)[0].cpu(),features=F.normalize(per[valid],dim=-1).half().cpu(),mass=mass[valid].cpu()),path);source_files.append(path)
            print(a.key,frame,'dense',rh,rw,'cells',int(valid.sum()),flush=True)
        del projection,radio
        features={mode:F.normalize(v,dim=-1).half().cpu() for mode,v in sums.items()};del sums
        robust=features['surface'].float().cuda()
        for _ in range(3):
            total=torch.zeros_like(robust);mass_total=torch.zeros(count,device='cuda')
            for path in source_files:
                record=load(path);ids=record['ids'].cuda();v=record['features'].float().cuda()
                distance=(2-2*(robust[ids]*v).sum(1)).clamp_min(1e-6).sqrt()
                weights=record['mass'].cuda()*(.25/distance).clamp_max(1)
                total.index_add_(0,ids,v*weights[:,None]);mass_total.index_add_(0,ids,weights)
            robust=F.normalize(total,dim=-1)
        features['surface_robust']=robust.half().cpu();masses['surface_robust']=masses['surface'];del robust
    cache=load(Path('output/odds_kernel_extent_next40')/a.key/'source_training.pt');q=F.normalize(cache['query'].cuda(),dim=-1)
    fallback=cache['base_logits'].cuda().sigmoid()
    model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.requires_grad_(False).eval();attach_kernel_native(model,data)
    with torch.no_grad():threshold=model.native_query_thresholds['text'](q)
    for mode,feat in features.items():
        ids=torch.arange(n,device='cuda') if mode.startswith('gaussian') else mapping
        supported=(masses[mode][ids]>0)&observed
        with torch.no_grad():cosine=torch.cat([F.normalize(part.float().cuda(),dim=-1)@q.T for part in feat.split(4096)])[ids]
        examples=[]
        for e in cache['examples']:
            if e['view'] not in cache['held']:examples.append({k:v.cuda() if torch.is_tensor(v) else v for k,v in e.items()})
        logscale=torch.nn.Parameter(torch.tensor(3.,device='cuda'));offset=torch.nn.Parameter(torch.tensor(0.,device='cuda'));opt=torch.optim.Adam([logscale,offset],lr=.02)
        for step in range(100):
            e=examples[step%len(examples)];col=e['columns'];value=torch.sigmoid(logscale.exp()*(cosine[:,col]-threshold[col]+offset))
            value=torch.where(supported[:,None],value,fallback[:,col])
            pred=torch.sparse.mm(e['op'],value);y=e['target'];known=e['known'];pos=known*y;neg=known*(1-y)
            loss=(((pred-y).square()*pos).sum()/pos.sum().clamp_min(1)+((pred-y).square()*neg).sum()/neg.sum().clamp_min(1))/2
            opt.zero_grad();loss.backward();opt.step()
        state=dict(features=feat,gaussian_to_element=ids.cpu(),supported=supported.cpu(),scale=float(logscale.exp()),offset=float(offset),voxel_size=size,normal_bins=6)
        state['local_language_head']=a.language_head
        state['native_vision']=a.native_vision
        dest=Path(a.output)/mode/a.key;dest.mkdir(parents=True,exist_ok=True);torch.save(dict(data,dense_surface_field=state),dest/'field.pt')
        (dest/'construction.json').write_text(json.dumps(dict(key=a.key,mode=mode,elements=len(feat),gaussians=n,supported=int(supported.sum()),scale=state['scale'],offset=state['offset'],voxel_size=size,feature_bytes=feat.numel()*2,source_frames=info['train_frames'],teacher=('native SigLIP2 vision patch tokens' if a.native_vision else 'official C-RADIOv4 siglip2-g spatial adaptor on complete cached grids'),local_language_head=a.language_head,local_patch_window=3 if a.language_head else None,geometry_changed=False,image_path_changed=False,source_calibration='100 fixed Adam steps, class-balanced known Brier, no benchmark queries',robust_iterations=3 if mode=='surface_robust' else 0),indent=2)+'\n')
        print(a.key,mode,'saved',flush=True)


if __name__=='__main__':main()
