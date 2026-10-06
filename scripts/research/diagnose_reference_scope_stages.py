"""Separate source-track reconstruction from independent target-mask accuracy.

This does not run tracking on target RGB or treat source masks as ground truth.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior


def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def metrics(p,t):
    p=np.asarray(p,dtype=bool);t=np.asarray(t,dtype=bool);tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum())
    return dict(iou=tp/max(1,tp+fp+fn),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),tp=tp,fp=fp,fn=fn)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4)
    root=Path('output/reference_integration');scene=json.loads(Path(f'configs/consolidated/nvos/{a.scene}.scene.json').read_text());query=json.loads(Path(f'configs/consolidated/nvos/{a.scene}.query.json').read_text());evaluation=json.loads(Path(f'configs/consolidated/nvos/{a.scene}.evaluation.json').read_text())
    response=load(root/'fresh'/a.scene/'field.pt');subset=load(root/'subset'/a.scene/'sets'/a.scene/'typed_readout.pt');tracks=load(root/'fresh_tracks'/a.scene/'tracks.pt');b=scene['geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda');gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);selected=((subset['selection_scores'][:,0]>.5)&subset['selection_observed']).cuda();values=response['elements'].cuda();observed=response['observed'].cuda();colors=torch.cat([values,observed.float()[:,None]],1)
    def render(frame,w,h):
        cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;kw=dict(viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False)
        image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,**kw);mass=image[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);r=torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0)
        _,sa,_=rasterization(means=geo.xyz[selected],quats=geo.quaternions[selected],scales=geo.scales[selected],opacities=geo.opacities.flatten()[selected],colors=torch.ones(int(selected.sum()),1,device='cuda'),**kw)
        return r.cpu().numpy(),sa[0,...,0].cpu().numpy()
    rows=[];h,w=scene['source_fit_hw']
    for frame in scene['source_frames']:
        rec=tracks['frames'].get(frame)
        if rec is None or rec['object_logit']<=0:continue
        r,alpha=render(frame,w,h);t=torch.nn.functional.interpolate(rec['logits'].float()[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0].numpy()>0
        row=dict(frame=frame,response_vs_track=metrics(r>=.5,t),subset_vs_track=metrics(alpha>10.5/255,t))
        if frame==query['reference_frame']:
            seed=np.array(Image.open(query['reference_mask']).convert('L').resize((w,h),Image.Resampling.NEAREST))>127;row['track_vs_input_seed']=metrics(t,seed)
        rows.append(row)
    w,h=evaluation['raster_wh'];r,alpha=render(evaluation['target_frame'],w,h);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'target_predictions.npz',response=r,subset_alpha=alpha)
    # Independent labels opened only after predictions are finalized.
    gt=np.array(Image.open(evaluation['gt_mask']));gt=np.any(gt>127,axis=2) if gt.ndim==3 else gt>127;size=(gt.shape[1],gt.shape[0]);r=cv2.resize(r,size,interpolation=cv2.INTER_LINEAR);alpha=cv2.resize(alpha,size,interpolation=cv2.INTER_LINEAR)
    result=dict(scene=a.scene,source_comparisons_are_pseudo_evidence=True,target_rgb_used=False,source=rows,source_macro={head:{m:float(np.mean([row[head][m] for row in rows])) for m in ['iou','precision','recall']} for head in ['response_vs_track','subset_vs_track']},target=dict(response=metrics(r>=evaluation['threshold'],gt),subset_diagnostic=metrics(alpha>10.5/255,gt)),independent_seed_or_track_accuracy_identifiable=False)
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,result['source_macro'],result['target'],flush=True)

if __name__=='__main__':main()
