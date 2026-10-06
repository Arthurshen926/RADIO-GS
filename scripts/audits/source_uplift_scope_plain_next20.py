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
    root=Path('output/full_field_rebuild_next19')/a.scene/'reconstruction/field';out=Path('output/source_uplift_scope_plain_next20')/a.scene;out.mkdir(parents=True,exist_ok=False)
    reg=dict(source_only=True,target_data_used=False,geometry_fixed=True,method='renderer-adjoint weighted source-mask lifting; not exact LUDVIG reproduction',validation='last four source train frames; not held out from field construction',confidence=.7,min_pixels=20,min_views=2,max_view_variance=.04,probability_interval=[.25,.75],logit_bound=2.197224577,unknown='exact baseline fallback',arms=['native','final','lifted','local'],parameter_search=False,source_query_representation='plain source grounding phrases, same encoder/storage as deployment; no template prefix',source_masks='SAM pseudo labels; outside-mask negatives are not verified truth')
    (out/'registration.json').write_text(json.dumps(reg,indent=2))
    fp=root/'bounded_fallback_admission/field.pt';fr=json.loads((fp.parent/'report.json').read_text());assert sha256_file(fp)==fr['field_sha256'];payload=load(fp)
    field=ElementRetrievalField(**payload['architecture']).cuda();field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
    meta=json.loads((root/'exact_render_query_threshold96/report.json').read_text());assert sha256_file(root/'source_grounded_text.pt')==meta['raw_source_grounding_sha256'];assert sha256_file(root/'episodes.pt')==meta['episode_sha256']
    raw=load(root/'source_grounded_text.pt');episode=load(root/'episodes.pt');info=json.loads((root/'teachers.json').read_text());assert not raw['benchmark_labels_used'];assert raw['frames']==info['train_frames'];assert raw['geometry_binding']==payload['geometry_binding']
    qids=meta['query_indices'];prototype_path=Path('output/source_uplift_scope_next20')/a.scene/'plain_queries.pt';prototype=load(prototype_path);assert prototype['query_ids']==qids;assert prototype['raw_source_sha256']==meta['raw_source_grounding_sha256'];q=prototype['queries'].cuda();Q=len(qids);N=len(field.latent);h,w=raw['probability'].shape[-2:];held=set(raw['frames'][-4:])
    gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json';gr=json.loads(gp.read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0');geo.assert_binding(payload['geometry_binding']);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    def kwargs(frame):
        cam=prior.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
        return dict(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
    with torch.no_grad():
        parent_path=Path('output/source_uplift_scope_next20')/a.scene/'uplift.pt';parent_report=json.loads((parent_path.parent/'report.json').read_text());assert sha256_file(parent_path)==parent_report['uplift_sha256'];parent=load(parent_path);assert parent['source_field_sha256']==fr['field_sha256'];assert parent['geometry_binding']==payload['geometry_binding'];mean=parent['mean'].cuda();variance=parent['variance'].cuda();count=parent['count'].cuda();denom=(count>0).float()
        native=torch.sigmoid(field.readout_log_scale['text'].clamp(-4,6).exp()*(field.native_cosine(q)-field.native_query_thresholds['text'](q)[None]))*field.observed[:,None]
        final=field.retrieve(q,'text')['scores'];lifted=torch.where(denom>1e-6,mean,final)*field.observed[:,None];local,reliable=local_correction(final,mean,variance,count);local*=field.observed[:,None]
        scores=dict(native=native,final=final,lifted=lifted,local=local);rows=[]
        for vi,frame in enumerate(raw['frames']):
            if frame not in held:continue
            for qi,qid in enumerate(qids):
                gt=raw['probability'][vi,qid].cuda()>.5
                if raw['scores'][vi,qid]<.7 or gt.sum()<20:continue
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
                rows.append(dict(frame=frame,query=raw['queries'][qid],query_index=qid,metrics=metrics))
            print(a.scene,'validation',frame,flush=True)
        macro={n:{m:float(np.mean([r['metrics'][n][m] for r in rows])) for m in ['probability_iou','selected_iou']} for n in scores};assert rows
        accepted={n:all(macro[n][m]>=macro['final'][m] for m in macro[n]) and any(macro[n][m]>macro['final'][m]+1e-6 for m in macro[n]) for n in ['lifted','local']}
        pack=dict(query_ids=qids,queries=q.cpu(),mean=mean.cpu(),variance=variance.cpu(),count=count.cpu(),geometry_binding=payload['geometry_binding'],source_field_sha256=fr['field_sha256'],train_frames=[f for f in raw['frames'] if f not in held],validation_frames=sorted(held))
        torch.save(pack,out/'uplift.pt')
        report=dict(complete=True,registration=reg,prototype_sha256=sha256_file(prototype_path),parent_uplift_sha256=sha256_file(parent_path),macro=macro,source_accepted=accepted,observations=rows,field_sha256=fr['field_sha256'],raw_source_sha256=meta['raw_source_grounding_sha256'],geometry_report_sha256=sha256_file(gp),implementation_sha256=sha256_file(__file__),uplift_sha256=sha256_file(out/'uplift.pt'),reliable_element_query_fraction=float(reliable.float().mean()),covered_element_query_fraction=float((denom>1e-6).float().mean()),unknown_fallback_exact=bool(torch.equal(local[~reliable],final[~reliable])),geometry_unchanged=True,field_unchanged=True)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,macro,accepted,flush=True)


if __name__=='__main__':main()
