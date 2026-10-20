"""Reference-identified source masks to a frozen field using one shared reference solver.

No scene name, historical query scores, target image or target labels determine
inference. Scene/query paths and source tracks are explicit inputs.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from radio_gs import evidence_core
import cv2,numpy as np
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.querying.typed_evidence import solve_rendered_evidence
from radio_gs.querying.instance_reliability import instance_reliability


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--query',required=True);p.add_argument('--tracks');p.add_argument('--evidence',help='Explicit per-view instance evidence; masks and unknown domain');p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=120);p.add_argument('--negative-policy',choices=['original','other-view-positive'],default='original');p.add_argument('--positive-core',choices=['eroded','medial','raw'],default='eroded');p.add_argument('--reference-only',action='store_true',help='Fixed evidence ablation: only query reference frame supplies masks');a=p.parse_args(argv)
    torch.set_num_threads(4);start=time.monotonic();scene=json.loads(Path(a.scene).read_text());query=json.loads(Path(a.query).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    local_reliability=scene.get('instance_evidence_reliability') in ('local','positive_local')
    tracks=torch.load(a.tracks,map_location='cpu',weights_only=False) if a.tracks else None;binding=scene['geometry'];geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda');gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256'])
    n=len(geo.xyz);h,w=scene['source_fit_hw'];radius=scene['boundary_radius'];observed=torch.zeros(n,dtype=torch.bool,device='cuda');observations=[];positive_terms=[];active=[];evidence=out/'evidence';evidence.mkdir(exist_ok=True)
    for frame in scene['source_frames']:
        cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
        renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
        hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False)
        pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();weight=hit['weights'].float().cuda();observed[gid[weight>1e-8]]=True
        total=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight)
        if a.evidence:
            explicit=torch.load(Path(a.evidence)/f'identity_{frame}.pt',map_location='cpu',weights_only=False)
            record=explicit['instances'][0]
            present=bool(len(record['scores'])) and (not a.reference_only or frame==query['reference_frame'])
        else:
            record=tracks['frames'].get(frame)
            present=record is not None and record['object_logit']>0 and (not a.reference_only or frame==query['reference_frame'])
        instance=dict(scores=torch.empty(0),masks=torch.empty(0,h,w))
        if present:
            op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/total[pid].clamp_min(1e-8),(h*w,n)).coalesce()
            if a.evidence:
                y=F.interpolate(record['masks'].float().amax(0)[None,None].cuda(),size=(h,w),mode='bilinear',align_corners=False)
            else:
                y=F.interpolate(record['logits'].float()[None,None].cuda(),size=(h,w),mode='bilinear',align_corners=False).sigmoid()
            binary=(y>.5).float()
            pos=(-F.max_pool2d(-binary,2*radius+1,1,radius)>.5).flatten()[:,None];neg=(F.max_pool2d(binary,2*radius+1,1,radius)<.5).flatten()[:,None]
            if a.positive_core=='raw':pos=binary.flatten()[:,None]>.5
            elif a.positive_core=='medial':
                distance=cv2.distanceTransform(binary[0,0].cpu().numpy().astype(np.uint8),cv2.DIST_L2,5)
                distance=torch.from_numpy(distance).to(binary.device)[None,None]
                ridge=(distance>0)&(distance>=F.max_pool2d(distance,3,1,1))
                pos=pos|ridge.flatten()[:,None]
            unknown=record.get('unknown_masks') if a.evidence else None
            if unknown is not None and len(unknown):
                unknown=F.interpolate(unknown.float().amax(0)[None,None].cuda(),size=(h,w),mode='bilinear',align_corners=False)[0,0].flatten()[:,None]>.5
                neg &= ~unknown
            known=(pos.float()+neg.float()*float(record.get('negative_reliability',1.)))*(total[:,None]>1e-8)
            if local_reliability:
                observation=(record if a.evidence else dict(scores=torch.as_tensor(record['object_logit']).float().sigmoid().reshape(1),
                    masks=y[0].cpu(),negative_reliability=record.get('negative_reliability',1.)))
                positive_weight,negative_weight=instance_reliability(observation,(h,w),y.device,
                    complement='view' if scene.get('instance_evidence_reliability')=='positive_local' else 'minimum')
                known=(pos*positive_weight.flatten()[:,None]+neg*negative_weight)*(total[:,None]>1e-8)
            if evidence_core.enabled():
                domains=evidence_core.compile_domains(pos, neg, visible=total[:,None]>1e-8,
                    negative_origin='model_complement')
                pw=positive_weight.flatten()[:,None] if local_reliability else 1.
                nw=negative_weight if local_reliability else float(record.get('negative_reliability',1.))
                positive, negative=evidence_core.signed_weights(domains, pw, nw)
                known=positive+negative
            observations.append((op,y.flatten()[:,None],known,1.));positive_terms.append((pos,neg));active.append(frame)
            instance=dict(scores=torch.ones(1),masks=y[0].cpu())
            if local_reliability:
                # Preserve the same instance reliability for actual-set fitting.
                # Explicit multi-instance input keeps its instances until fusion.
                instance=dict(observation,masks=observation['masks'].cpu())
            if unknown is not None and len(unknown):instance['unknown_masks']=unknown.reshape(1,h,w).cpu()
        torch.save(dict(queries=[query['id']],instances=[instance],frame=frame,provider='explicit reference-instance temporal masks'),evidence/f'identity_{frame}.pt')
    protection_summary={}
    if a.negative_policy=='other-view-positive':
        positive_mass=[];known_mass=[]
        for (op,y,known,weight),(pos,neg) in zip(observations,positive_terms):
            positive_mass.append(torch.sparse.mm(op.T,pos.float()));known_mass.append(torch.sparse.mm(op.T,known))
        sum_pos=sum(positive_mass);sum_known=sum(known_mass);updated=[];before=0.;after=0.
        for i,((op,y,known,weight),(pos,neg)) in enumerate(zip(observations,positive_terms)):
            other_mass=(sum_known-known_mass[i]).clamp_min(0);other_pos=(sum_pos-positive_mass[i]).clamp_min(0)
            other=other_pos/other_mass.clamp_min(1e-8);coverage=torch.sparse.mm(op,(other_mass>1e-8).float())
            support=torch.sparse.mm(op,other)/coverage.clamp_min(1e-8);support=torch.where(coverage>.5,support.clamp(0,1),0)
            reliability=torch.where(neg,1-support,torch.ones_like(support));before+=float((known*neg).sum());new_known=known*reliability;after+=float((new_known*neg).sum());updated.append((op,y,new_known,weight))
        observations=updated;protection_summary=dict(negative_mass_before=before,negative_mass_after=after,own_view_excluded=True)
    value,mass=solve_rendered_evidence(torch.full((n,1),.5,device='cuda'),observations,steps=a.steps)
    value[(~observed)[:,None]|(mass<=1e-8)]=0
    torch.save(dict(elements=value.cpu(),observed=observed.cpu(),geometry=binding),out/'field.pt')
    # Zero no-evidence support initialization for a separate selected-set control, not a native semantic feature.
    torch.save(dict(texts=[query['id']],semantic_probability=torch.zeros(n,1),observed=observed.cpu(),geometry=binding),out/'neutral_support.pt')
    (out/'subset_inputs.json').write_text(json.dumps(dict(scene=a.scene,response=str(out/'neutral_support.pt'),detections=str(evidence),tracks=[None],base_raster=[h,w],evidence_contract='reference_instance; neutral initialization; no text retrieval or historical query prediction'),indent=2)+'\n')
    (out/'fit.json').write_text(json.dumps(dict(scene=a.scene,query=a.query,tracks=a.tracks,explicit_evidence=a.evidence,source_frames=scene['source_frames'],active_frames=active,reference_only=a.reference_only,positive_core=a.positive_core,negative_policy=a.negative_policy,protection=protection_summary,observed_elements=int(observed.sum()),steps=a.steps,source_fit_hw=[h,w],boundary_radius=radius,solver='radio_gs.querying.typed_evidence.solve_rendered_evidence',target_rgb_used=False,target_gt_used=False,seconds=time.monotonic()-start),indent=2)+'\n')


if __name__=='__main__':main()
