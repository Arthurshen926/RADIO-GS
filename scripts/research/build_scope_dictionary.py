"""Construct a multi-granularity native field from existing source evidence."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.native_scope_dictionary import spherical_dictionary, NativeScopeDictionary, pixel_scope_owners
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.training.source_region_readback import region_render_weights


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--output',default='output/scope_dictionary_next33');p.add_argument('--granularity',choices=['physical','hierarchy'],default='physical');p.add_argument('--query-code-routing',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    rows=json.loads(Path('output/native_residual_basis_next30_v2/registration.json').read_text())['cohort']
    row=next(r for r in rows if r['key']==a.key);root=Path(row['root']);out=Path(a.output)/a.key;out.mkdir(parents=True,exist_ok=True)
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    teacher=load(root/info['teacher_files']['siglip']['path']);ep=load(root/'episodes.pt')
    parent=load(root/row['run']/'field.pt');observed=parent['state_dict']['observed'];n=len(observed)
    native=F.normalize(teacher['descriptors'].float(),dim=1);masks=ep['masks'];views=ep['proposal_views']
    extra=Path('output/small_region_restoration_next32')/a.key/'small_teacher.pt'
    if (out/'small_teacher.pt').exists():extra=out/'small_teacher.pt'
    if extra.exists():
        small=load(extra);native=torch.cat([native,F.normalize(small['descriptors'].float(),dim=1)])
        masks=torch.cat([masks,small['masks']]);views=torch.cat([views,small['proposal_views']])
    native=native.cuda()
    training=torch.tensor([info['frames'][int(vi)] in info['train_frames'] for vi in views])
    dictionary,_=spherical_dictionary(native[training]);codes=(native@dictionary.T).argmax(1)
    quantization=(native[training]*dictionary[codes[training]]).sum(1)
    if row['benchmark']=='lerf':
        gp=Path('output/depth_opacity_repair_20260918')/row['scene']/'geometry_report.json'
    elif row['benchmark']=='nvos':
        gp=root.parent/'frozen_geometry/report.json'
    else:
        gp=root.parent.parent/'geometry_report.json'
    gr=json.loads(gp.read_text())['results']['independent_scale']
    geometry=load(gr['geometry'])
    xyz=geometry['state_dict']['xyz'].float().cuda()
    radii=torch.zeros(len(masks),device='cuda')
    def view_data(vi):
        record=auth['views'][vi];hit=load(root/record['relative_path'])
        return hit['pixel_ids'].long().cuda(),hit['gaussian_ids'].long().cuda(),hit['base_weights'].float().cuda()
    # Physical radius is view-independent in units; bins are fitted on source views.
    for frame in (info['train_frames'] if a.granularity=='physical' else []):
        vi=info['frames'].index(frame);ids=torch.where(views==vi)[0];pid,gid,w=view_data(vi)
        lift,_=source_pixel_evidence_weights(masks[ids].flatten(1).cuda(),pid,gid,w,n,partition_overlap=False)
        den=lift.sum(1).clamp_min(1e-8);center=lift@xyz/den[:,None]
        variance=lift@xyz.square().sum(1)/den-center.square().sum(1)
        radii[ids]=variance.clamp_min(1e-10).sqrt()
    boundaries=(torch.quantile(radii[training].log(),torch.tensor([1/3,2/3],device='cuda')) if a.granularity=='physical' else torch.zeros(2,device='cuda'))
    levels=torch.bucketize(radii.log(),boundaries)
    scale_counts=torch.zeros((len(dictionary),3),device='cuda')
    if a.granularity=='physical':scale_counts.index_put_((codes[training],levels[training]),torch.ones(int(training.sum()),device='cuda'),accumulate=True)
    torch.save(dict(levels=levels.cpu()),out/'training.pt')
    histogram=torch.zeros((3,n,len(dictionary)),device='cuda')
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);ids=torch.where(views==vi)[0];pid,gid,w=view_data(vi)
        if a.granularity=='hierarchy':
            owners,covered=pixel_scope_owners(masks[ids].flatten(1).cuda())
            mass=torch.zeros(masks.shape[-2]*masks.shape[-1],device='cuda').scatter_add_(0,pid,w)
            operator=torch.sparse_coo_tensor(torch.stack([gid,pid]),w/mass[pid].clamp_min(1e-8),(n,len(covered))).coalesce()
            for level in range(3):
                pixel_codes=codes[ids.cuda()[owners[level]]]
                colors=F.one_hot(pixel_codes,len(dictionary)).float()*covered[:,None]
                histogram[level]+=torch.sparse.mm(operator,colors)
                scale_counts[:,level].scatter_add_(0,pixel_codes,covered.float())
            print(a.key,'hierarchy fused',frame,flush=True)
            continue
        for level in range(3):
            chosen=ids[levels[ids].cpu()==level]
            if not len(chosen):continue
            lift,_=source_pixel_evidence_weights(masks[chosen].flatten(1).cuda(),pid,gid,w,n,partition_overlap=True)
            histogram[level].index_add_(1,codes[chosen],lift.T)
        print(a.key,'fused',frame,flush=True)
    values,indices=histogram.topk(4,dim=2)
    retained_mass=values.sum(2)/histogram.sum(2).clamp_min(1e-8)
    coverage=histogram.sum(2)>0
    weights=values/values.sum(2,keepdim=True).clamp_min(1e-8)
    scale_prior=scale_counts/scale_counts.sum(1,keepdim=True).clamp_min(1)
    payload=dict(codebook=dictionary.cpu().half(),indices=indices.permute(1,0,2).cpu().short(),
        weights=weights.permute(1,0,2).cpu().half(),observed=observed,geometry_binding=parent['geometry_binding'],
        scale_boundaries=boundaries.exp().cpu(),scale_prior=scale_prior.cpu(),temperature=20.,root=str(root),geometry_report=str(gp),
        granularity=a.granularity,query_code_routing=a.query_code_routing,native_modalities=['image'] if row['benchmark']=='nvos' else ['text'],method='3 granularity distributions;512 native codewords;4 entries per element and scale',
        base_field=str(root/row['run']/'field.pt'))
    torch.save(payload,out/'field.pt')
    construction=dict(key=a.key,training_regions=int(training.sum()),dictionary_size=len(dictionary),
        native_quantization_cosine_mean=float(quantization.mean()),physical_scale_boundaries=boundaries.exp().tolist(),
        sparse_mass_retained=float(retained_mass[coverage].mean()),extra_per_element_bytes=48,
        source_views=len(info['train_frames']),new_small_regions=int(len(masks)-len(ep['masks'])),
        mainline_replaced=False)
    del histogram,values,indices,geometry,xyz
    torch.cuda.empty_cache()
    field=NativeScopeDictionary(payload)
    train_native=native[training]
    queries=dict(text=F.normalize(teacher['text_embeddings'].float().cuda(),dim=1),
        image=train_native[torch.linspace(0,len(train_native)-1,min(256,len(train_native))).long()])
    baseline=F.normalize(parent['state_dict']['latent'][:,:64].cuda(),dim=1)
    basis=parent['state_dict']['native_decoder.weight'].float().cuda()
    metrics={}
    for modality,query in queries.items():
        predictions=dict(native64=baseline@(query@basis).T,dictionary=field.scores(query),routed_dictionary=field.scores(query,pool='routed'))
        rows={name:[] for name in predictions};targets=[]
        for frame in info['validation_frames']:
            vi=info['frames'].index(frame);ids=torch.where(views==vi)[0];pid,gid,w=view_data(vi)
            pooling,supported,_=region_render_weights(masks[ids].flatten(1).cuda(),pid,gid,w,observed.cuda())
            targets.append((native[ids]@query.T)[supported])
            for name,scores in predictions.items():rows[name].append((pooling@scores)[supported])
        target=torch.cat(targets)
        metrics[modality]={}
        for name,items in rows.items():
            pred=torch.cat(items);error=pred-target
            metrics[modality][name]=dict(mse=float(error.square().mean()),centered_mse=float((error-error.mean(1,keepdim=True)).square().mean()),top1=float((pred.argmax(1)==target.argmax(1)).float().mean()))
    construction['source_validation']=metrics
    (out/'report.json').write_text(json.dumps(construction,indent=2)+'\n')
    print(json.dumps(construction,indent=2),flush=True)


if __name__=='__main__':main()
