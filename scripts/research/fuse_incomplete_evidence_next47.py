"""One set-membership solver for text detections and reference video masks."""
import argparse,json,shutil,time
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.querying.typed_evidence import SetType,visible_operator,view_evidence,solve_set


def load(path):return torch.load(path,map_location='cpu',weights_only=False)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--modes',default='plain,identity_only,consistency_only,both,no_prior,censored');p.add_argument('--tracks-base',default='output/source_video_next43');p.add_argument('--key');p.add_argument('--scene');a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
    reference=a.scene is not None
    key='nvos_'+('horns_center' if a.scene=='horns_left' else a.scene) if reference else a.key
    task=SetType.REFERENCE_INSTANCE if reference else SetType.TEXT_UNION
    parent=Path('output/taskwise_scope_composition_next38')/key;data=load(parent/'field.pt');root=Path(data['root'])
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());state=load(data['base_field'])['state_dict']
    observed=(state.get('image_observed',state['observed']) if reference else state['observed']).cuda();h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
    if reference:
        prior=load(parent/('evaluation_'+a.scene)/'trained.pt')['elements'].float().cuda()
        tracking=load(Path(a.tracks_base)/a.scene/'tracks.pt');frames=[f for f,r in tracking['frames'].items() if f!=tracking['reference_frame'] and f in info['train_frames'] and r['object_logit']>0]
    else:
        previous=load(parent/'anchored/pipeline_calibrated/scores.pt');original=previous['scores']['hierarchical_pipeline'].cuda();categorical=key.startswith('scannet_');prior=original.sigmoid() if categorical else original;frames=info['train_frames']
    for mode in (a.modes.split(',') if a.modes else (['corroborated'] if reference else ['direct','corroborated'])):
        examples=[]
        if mode in ['identity_only','both','no_prior','censored','matched_confidence']:
            identity_cache={f:load(Path('output/typed_evidence_next44')/key/f'identity_{f}.pt') for f in frames}
            square=torch.zeros(prior.shape[1]);count=torch.zeros_like(square)
            for d in identity_cache.values():
                for j,r in enumerate(d['instances']):square[j]+=r['foreground_margin'].square().sum();count[j]+=len(r['scores'])
            margin_scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6)
        if reference and mode=='full_video':
            frames=[f for f,r in tracking['frames'].items() if f!=tracking['reference_frame'] and r['object_logit']>0]
        if reference and mode=='bridge':
            frames=[f for f,r in tracking['frames'].items() if f!=tracking['reference_frame'] and f in info['train_frames'] and r['object_logit']>0]
        for frame in frames:
            identity_confidence=None
            if frame in info['frames']:
                hit=load(root/auth['views'][info['frames'].index(frame)]['relative_path'])
            else:
                cache=Path(a.tracks_base)/a.scene/f'geometry_{frame}.pt'
                if cache.exists():hit=load(cache)
                else:
                    from types import SimpleNamespace
                    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
                    from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
                    from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
                    gr=json.loads(Path(data['geometry_report']).read_text());g=gr['results']['independent_scale']
                    geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda');geometry_prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);camera=geometry_prior.cameras[frame]
                    k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
                    renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*geometry_prior.scale,far_plane=100*geometry_prior.scale)
                    hits=rasterize_single_view_contributions(geo,renderer,camera.world_to_camera.cuda(),height=h,width=w,include_depths=False)
                    hit=dict(pixel_ids=hits['pixel_ids'].int().cpu(),gaussian_ids=hits['gaussian_ids'].int().cpu(),base_weights=hits['weights'].cpu());torch.save(hit,cache)
                    del geo,hits,geometry_prior

            op,evidence_op,visible=visible_operator(hit,observed,h*w,ownership=mode=='corroborated')
            if reference:
                logits=tracking['frames'][frame]['logits'].float().cuda()
                probability=F.interpolate(logits[None,None],size=(h,w),mode='bilinear',align_corners=False).sigmoid().flatten()[:,None];detected=torch.ones(1,dtype=torch.bool,device='cuda')
            else:
                detection=load(Path('output/typed_evidence_next44')/key/(f'identity_{frame}.pt' if mode=='legacy_hard_identity' else f'detection_{frame}.pt'))
                if mode in ['identity_only','both','no_prior','censored','matched_confidence']:
                    detection=identity_cache[frame]
                    identity_confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/margin_scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(detection['instances'])],device='cuda')
                if mode=='legacy_hard_identity':
                    detection['instances']=[dict(masks=r['masks'][r['foreground_margin']>0],scores=r['scores'][r['foreground_margin']>0]) for r in detection['instances']]
                if mode in ['detector_only','detector_consistency']:
                    identity_confidence=torch.tensor([float(r['scores'].max()) if len(r['scores']) else 0. for r in detection['instances']],device='cuda')
                # Preserve per-instance observations in the cache. Only a text
                # union explicitly permits union here; no cross-view ID merge.
                probability=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in detection['instances']],dim=-1).reshape(h*w,-1).cuda()
                detected=torch.tensor([len(r['scores'])>0 for r in detection['instances']],device='cuda')
            examples.append(view_evidence(op,evidence_op,visible,probability,detected,frame,h,w,confidence=identity_confidence))
        if mode=='matched_confidence':
            detector=torch.stack([torch.tensor([float(r['scores'].max()) if len(r['scores']) else 0. for r in identity_cache[e.frame]['instances']],device='cuda') for e in examples])
            area=torch.stack([(e.positive|e.negative).float().sum(0) for e in examples]);actual=torch.cat([e.confidence for e in examples],0)
            constant=(actual*area).sum(0)/(detector*area).sum(0).clamp_min(1e-8)
            for index,e in enumerate(examples):e.confidence=(detector[index]*constant)[None]
        initial=torch.where(observed[:,None],torch.full_like(prior,.5),torch.zeros_like(prior)) if mode=='no_prior' else prior
        value,stats=solve_set(initial,examples,task,corroborate=False,consistency=mode in ['consistency_only','detector_consistency','both','no_prior','censored','matched_confidence'],prior_strength=0. if mode=='no_prior' else .1,censor_positive=mode=='censored')
        out=Path('output/incomplete_evidence_next47')/mode/(a.scene if reference else key);out.mkdir(parents=True,exist_ok=True)
        if reference:torch.save(dict(elements=value.cpu()),out/'elements.pt')
        else:
            prediction=torch.where(value!=prior,torch.logit(value.clamp(1e-6,1-1e-6)),original) if categorical else value
            torch.save(dict(texts=previous['texts'],scores=prediction.cpu()),out/'elements.pt')
            shutil.copyfile(parent/'field.pt',out/'field.pt');shutil.copyfile(parent/'calibration.pt',out/'calibration.pt')
        stats.update(key=key,scene=a.scene,mode=mode,seconds=time.monotonic()-start,target_rgb_used=False,query_conditioned_online=True,independent_identity='reference-seeded video' if reference else 'SAM3 text detector',identity_foreground_context=mode in ['identity_only','both','no_prior','censored','matched_confidence'],geometry_changed=False,semantic_vectors_written=False,forward='full frozen renderer',update='visibility-restricted' if mode=='corroborated' else 'full responsibility')
        (out/'method.json').write_text(json.dumps(stats,indent=2)+'\n');print(stats,flush=True)

if __name__=='__main__':main()
