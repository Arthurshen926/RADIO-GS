"""Shared frozen LERF scoring body extracted without changing rendering or thresholds."""
import json
from pathlib import Path
import numpy as np
import torch


@torch.inference_mode()
def evaluate_lerf_readouts(*, scene, out, texts, scores, selection_scores,
                          geo, default_observed, semantic_geo, semantic_observed,
                          selection_geo, selection_observed, camera_report,
                          camera_report_sha256, pipeline=True,
                          label_root='/mnt/pool/sqy/3d_understanding/lerf_ovs/label',
                          selected_gt_root='output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/eval_gt',
                          score_selected=True, report_writer=None, comparison_reference=None):
    from gsplat import rasterization
    from sklearn.metrics import average_precision_score
    from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
    from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects,_localization_hit,_mask_iou
    from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES,evaluate_scene
    from PIL import Image
    from dataclasses import asdict
    prior=MogeGeometryPrior(camera_report,camera_report_sha256)
    frames=load_lerf_objects(Path(label_root)/scene,frames=SCENE_GT_FRAMES[scene])
    rows={name:[] for name in scores}
    if comparison_reference is not None and comparison_reference not in scores:
        raise ValueError('comparison reference must be one of the supplied responses')
    names=list(scores)
    if comparison_reference is not None:
        names.remove(comparison_reference); names.insert(0,comparison_reference)
    for frame,objects in frames.items():
        h,w=objects[0].mask.shape;cam=prior.cameras[int(frame.split('_')[-1])];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
        reference_masks={}
        for name in names:
            values=scores[name]
            for obj in objects:
                response_geo=semantic_geo if name=='hierarchical_pipeline' else geo
                response_observed=semantic_observed if name=='hierarchical_pipeline' else default_observed
                colors=torch.stack([values[:,texts.index(obj.query)]*response_observed,response_observed.float()],1)
                rendered,alpha,_=rasterization(means=response_geo.xyz,quats=response_geo.quaternions,scales=response_geo.scales,opacities=response_geo.opacities.flatten(),colors=colors,viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
                mass=rendered[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
                result=torch.where(valid,rendered[0,...,0]/mass.clamp_min(1e-8),0 if pipeline else -1).cpu()
                record=dict(frame=frame,query=obj.query,ap=float(average_precision_score(obj.mask.flatten(),result.numpy().flatten())),localization_hit=_localization_hit(torch.nonzero(result==result.max()),obj.bboxes))
                if pipeline:
                    record['iou']=_mask_iou(torch.from_numpy(obj.mask),result>.5)
                    if comparison_reference is not None:
                        binary=result>.5; truth=torch.from_numpy(obj.mask).bool()
                        if name==comparison_reference:reference_masks[id(obj)]=binary
                        else:
                            baseline=reference_masks[id(obj)]
                            record.update(corrected_pixels=int(((binary==truth)&(baseline!=truth)).sum()),
                                          harmed_pixels=int(((binary!=truth)&(baseline==truth)).sum()))
                if pipeline and score_selected:
                    set_geo=selection_geo if name=='hierarchical_pipeline' else geo
                    set_observed=selection_observed if name=='hierarchical_pipeline' else default_observed
                    selected=(selection_scores[name][:,texts.index(obj.query)]>.5)&set_observed
                    if selected.any():
                        _,mask,_=rasterization(means=set_geo.xyz[selected],quats=set_geo.quaternions[selected],scales=set_geo.scales[selected],opacities=set_geo.opacities.flatten()[selected],colors=torch.ones(int(selected.sum()),1,device='cuda'),viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
                        raster=(mask[0,...,0]*255+.5).floor().clamp(0,255).byte().cpu().numpy()
                    else:raster=np.zeros((h,w),dtype=np.uint8)
                    selected_dir=out/name;selected_dir.mkdir(exist_ok=True)
                    Image.fromarray(raster).save(selected_dir/(frame+'_'+obj.query+'.png'))
                rows[name].append(record)
        print(scene,frame,flush=True)
    metrics={name:dict(ap=float(np.mean([r['ap'] for r in values])),localization=float(np.mean([r['localization_hit'] for r in values])),observations=len(values)) for name,values in rows.items()}
    if pipeline:
        for name,values in rows.items():
            metrics[name]['iou']=float(np.mean([r['iou'] for r in values]))
            if score_selected:
                metrics[name]['selected']=asdict(evaluate_scene(Path(selected_gt_root)/scene/'gt',out/name,scene,threshold=10))
    if report_writer is None:
        (out/'query_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    else:
        report_writer('query_results.json',rows)
    return metrics
