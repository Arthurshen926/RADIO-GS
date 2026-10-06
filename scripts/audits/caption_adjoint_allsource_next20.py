"""Renderer-adjoint source scope control inspired by LUDVIG weighted uplifting.

Source masks only. No target data. Uses last four source frames for decoder
validation (the field itself saw these frames). No parent tensor mutation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from gsplat import rasterization
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


def local_correction(base,mean,variance,count):
    """Unknown/conflicting evidence leaves base EXACT; bounded local logit update."""
    reliable=(count>=2)&(variance<=.04)&((mean>=.75)|(mean<=.25))
    delta=(torch.logit(mean.clamp(.1,.9))).clamp(-2.197224577,2.197224577)
    return torch.where(reliable,torch.sigmoid(torch.logit(base.clamp(1e-6,1-1e-6))+delta),base),reliable


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
    root=Path('output/full_field_rebuild_next19')/a.scene/'reconstruction/field';out=Path('output/caption_adjoint_allsource_next20')/a.scene;out.mkdir(parents=True,exist_ok=False)
    reg=dict(source_layer='trusted-concept decoder; all source frames matched to parent; diagnostic masks are in-sample, not validation',source_only=True,target_data_used=False,geometry_fixed=True,method='renderer-adjoint caption concept union; original concept grouping unchanged; isolate trusted-concept decoder, not exact LUDVIG reproduction',validation='last four source frames are in-sample diagnostics; no source-generalization acceptance',confidence=.7,min_pixels=20,min_views=2,max_view_variance=.04,probability_interval=[.25,.75],logit_bound=2.197224577,unknown='exact baseline fallback',arms=['native','final','lifted','local'],parameter_search=False,source_masks='SAM pseudo labels; outside-mask negatives are not verified truth')
    (out/'registration.json').write_text(json.dumps(reg,indent=2))
    fp=root/'bounded_fallback_admission/field.pt';fr=json.loads((fp.parent/'report.json').read_text());assert sha256_file(fp)==fr['field_sha256'];payload=load(fp)
    field=ElementRetrievalField(**payload['architecture']).cuda();field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
    groups_report=json.loads((root/'grounded_semantic_transfer96/report.json').read_text());groups=groups_report['groups']
    rawpath=root/'caption_grounded_regions.pt';receipt=json.loads(rawpath.with_suffix('.json').read_text());assert sha256_file(rawpath)==receipt['sha256'];raw=load(rawpath)
    info=json.loads((root/'teachers.json').read_text());assert not raw['benchmark_data_used'];assert raw['teacher_manifest_sha256']==sha256_file(root/'teachers.json');assert raw['geometry_binding']==payload['geometry_binding']
    assert len(groups)==len(field.grounded_transfer.prototypes)
    assert all(raw['phrases'][g['source_query_index']]==g['phrase'] for g in groups)
    q=field.grounded_transfer.prototypes.detach();qids=list(range(len(groups)));Q=len(groups);N=len(field.latent);h,w=raw['masks'].shape[-2:];frames=info['train_frames'];held=set(frames[-4:])
    union={}
    for frame in frames:
        vi=info['frames'].index(frame);masks=[];eligible=[]
        for group in groups:
            members=[r for r in group['regions'] if int(raw['proposal_views'][r])==vi]
            masks.append(raw['masks'][members].any(0) if members else torch.zeros(h,w,dtype=torch.bool));eligible.append(bool(members))
        union[frame]=(torch.stack(masks),torch.tensor(eligible))
    gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json';gr=json.loads(gp.read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0');geo.assert_binding(payload['geometry_binding']);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    def kwargs(frame):
        cam=prior.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
        return dict(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
    numerator=torch.zeros(N,Q,device='cuda');denom=torch.zeros_like(numerator);second=torch.zeros_like(numerator);count=torch.zeros(N,Q,device='cuda',dtype=torch.int16)
    for vi,frame in enumerate(frames):
        # All source masks: match the complete parent decoder input budget.
        mask=union[frame][0].float().cuda();eligible=union[frame][1].cuda()
        # The renderer is linear in colors at fixed geometry: its color adjoint
        # is the exact alpha/transmittance-weighted pixel-to-element operator.
        for start in range(0,Q,8):
            stop=min(Q,start+8);use=eligible[start:stop]
            if not use.any():continue
            C=stop-start;colors=torch.zeros(N,C*2,device='cuda',requires_grad=True)
            im,_,_=rasterization(colors=colors,**kwargs(frame))
            target=mask[start:stop].permute(1,2,0)*use
            rhs=torch.cat([target,torch.ones_like(target)*use],-1)
            grad=torch.autograd.grad((im[0]*rhs).sum(),colors)[0].detach()
            n=grad[:,:C];d=grad[:,C:];valid=(d>1e-6)&field.observed[:,None];n*=valid;d*=valid
            per=n/d.clamp_min(1e-8)
            numerator[:,start:stop]+=n;denom[:,start:stop]+=d;second[:,start:stop]+=d*per.square();count[:,start:stop]+=valid.to(torch.int16)
            del colors,im,grad
        print(a.scene,'uplift',frame,flush=True)
    with torch.no_grad():
        mean=numerator/denom.clamp_min(1e-8);variance=(second/denom.clamp_min(1e-8)-mean.square()).clamp_min(0)
        head=field.grounded_transfer;start=head.identity_selection['trusted_concept_start'];assert start+Q==head.decoder.out_features
        parts=[]
        for z in field.latent.split(4096):
            features=head.features(z);v=features@head.decoder.weight[start:].T+head.decoder.bias[start:]
            if head.residual is not None:
                hidden=head.residual[1](head.residual[0](features));v+=hidden@head.residual[2].weight[start:].T+head.residual[2].bias[start:]
            parts.append(v.clamp(1e-6,1-1e-6))
        final=torch.cat(parts)*field.observed[:,None];native=final
        lifted=torch.where(denom>1e-6,mean,final)*field.observed[:,None];local,reliable=local_correction(final,mean,variance,count);local*=field.observed[:,None]
        scores=dict(final=final,lifted=lifted,local=local);rows=[]
        for vi,frame in enumerate(frames):
            if frame not in held:continue
            for qi,qid in enumerate(qids):
                gt=union[frame][0][qid].cuda()
                if not union[frame][1][qid] or gt.sum()<20:continue
                names=list(scores);colors=torch.stack([scores[n][:,qi] for n in names]+[field.observed.float()],1)
                im,alpha,_=rasterization(colors=colors,**kwargs(frame));mass=im[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);prob=torch.where(valid[...,None],im[0,...,:-1]/mass[...,None].clamp_min(1e-8),0);metrics={}
                for j,name in enumerate(names):
                    ids=(scores[name][:,qi]>.5)&field.observed;kw=kwargs(frame)
                    if ids.any():
                        for key in ['means','quats','scales','opacities']:kw[key]=kw[key][ids]
                        _,aa,_=rasterization(colors=torch.ones(int(ids.sum()),1,device='cuda'),**kw);selected=(aa[0,...,0]*255+.5).floor()>10
                    else:selected=torch.zeros_like(gt)
                    def iou(pred):return float((pred&gt).sum()/(pred|gt).sum().clamp_min(1))
                    metrics[name]=dict(probability_iou=iou(prob[...,j]>.5),selected_iou=iou(selected))
                rows.append(dict(frame=frame,query=groups[qid]['phrase'],query_index=qid,metrics=metrics))
            print(a.scene,'validation',frame,flush=True)
        macro={n:{m:float(np.mean([r['metrics'][n][m] for r in rows])) for m in ['probability_iou','selected_iou']} for n in scores};assert rows
        accepted={n:all(macro[n][m]>=macro['final'][m] for m in macro[n]) and any(macro[n][m]>macro['final'][m]+1e-6 for m in macro[n]) for n in ['lifted','local']}
        pack=dict(query_ids=qids,queries=q.cpu(),mean=mean.cpu(),variance=variance.cpu(),count=count.cpu(),geometry_binding=payload['geometry_binding'],source_field_sha256=fr['field_sha256'],train_frames=list(frames),validation_frames=sorted(held))
        torch.save(pack,out/'uplift.pt')
        report=dict(complete=True,registration=reg,macro=macro,source_accepted={'lifted':False,'local':False},in_sample_nonregression=accepted,observations=rows,field_sha256=fr['field_sha256'],raw_source_sha256=receipt['sha256'],groups_report_sha256=sha256_file(root/'grounded_semantic_transfer96/report.json'),geometry_report_sha256=sha256_file(gp),implementation_sha256=sha256_file(__file__),uplift_sha256=sha256_file(out/'uplift.pt'),reliable_element_query_fraction=float(reliable.float().mean()),covered_element_query_fraction=float((denom>1e-6).float().mean()),unknown_fallback_exact=bool(torch.equal(local[~reliable],final[~reliable])),geometry_unchanged=True,field_unchanged=True)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,macro,accepted,flush=True)


if __name__=='__main__':main()
