"""Source-only evidence tracing, then frozen target-label error attribution.

Witnesses are teacher positives, not verified object identity. Diagnostic only.
"""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4)
    scene=json.loads(Path(f'configs/consolidated/nvos/{a.scene}.scene.json').read_text());ev=json.loads(Path(f'configs/consolidated/nvos/{a.scene}.evaluation.json').read_text());root=Path('output/reference_integration');field=torch.load(root/'fresh'/a.scene/'field.pt',map_location='cpu',weights_only=False);tracks=torch.load(root/'fresh_tracks'/a.scene/'tracks.pt',map_location='cpu',weights_only=False)
    b=scene['geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda');gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);n=len(geo.xyz);h,w=scene['source_fit_hw'];radius=scene['boundary_radius'];stats=torch.zeros(n,4,device='cuda')
    for frame in scene['source_frames']:
        rec=tracks['frames'].get(frame)
        if rec is None or rec['object_logit']<=0:continue
        cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
        hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['weights'].float();total=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight);weight=weight/total[pid].clamp_min(1e-8)
        y=F.interpolate(rec['logits'].float()[None,None].cuda(),size=(h,w),mode='bilinear',align_corners=False);binary=(y>0).float();pos=(-F.max_pool2d(-binary,2*radius+1,1,radius)>.5).flatten();neg=(F.max_pool2d(binary,2*radius+1,1,radius)<.5).flatten();raw=binary.flatten()
        for i,v in enumerate([raw,pos.float(),neg.float(),(pos|neg).float()]):stats[:,i].scatter_add_(0,gid,weight*v[pid])
    observed=field['observed'].cuda();values=field['elements'].cuda();positive=stats[:,1]>1e-8;rawpositive=stats[:,0]>1e-8;known=stats[:,3]>1e-8;ratio=stats[:,1]/(stats[:,1]+stats[:,2]).clamp_min(1e-8)
    colors=torch.cat([values,observed[:,None].float(),rawpositive[:,None].float(),positive[:,None].float(),known[:,None].float(),ratio[:,None]],1)
    w,h=ev['raster_wh'];cam=gp.cameras[ev['target_frame']];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
    image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False)
    mass=image[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);response=torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0);fractions=image[0,...,1:]/alpha[0].clamp_min(1e-8)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);arrays=dict(response=response.cpu().numpy(),alpha=alpha[0,...,0].cpu().numpy(),fractions=fractions.cpu().numpy());np.savez_compressed(out/'target_evidence.npz',**arrays)
    # All source statistics and target projections fixed before GT access.
    gt=np.array(Image.open(ev['gt_mask']));gt=np.any(gt>127,axis=2) if gt.ndim==3 else gt>127;size=(gt.shape[1],gt.shape[0]);r=cv2.resize(arrays['response'],size);opacity=cv2.resize(arrays['alpha'],size);f=cv2.resize(arrays['fractions'],size);fn=gt&(r<ev['threshold']);remaining=fn.copy();parts={}
    conditions=[('no_rendered_geometry',opacity<=1e-8),('insufficient_observed_support',f[...,0]<=.5),('no_raw_positive_witness',f[...,1]<=1e-8),('positive_removed_by_erosion',f[...,2]<=1e-8),('positive_support_minority',f[...,2]<=.5),('positive_supported_but_response_below_threshold',np.ones_like(gt))]
    for name,condition in conditions:
        group=remaining&condition;parts[name]=dict(pixels=int(group.sum()),fraction_of_fn=float(group.sum()/max(1,fn.sum())));remaining&=~group
    result=dict(scene=a.scene,target_positive_pixels=int(gt.sum()),false_negative_pixels=int(fn.sum()),recall=float(1-fn.sum()/max(1,gt.sum())),fn_partition=parts,threshold_sensitivity_geometry_coverage={str(t):float((gt&(opacity>t)).sum()/max(1,gt.sum())) for t in [1e-8,.01,.1,.5]},positive_witness_fraction_quantiles_on_fn=np.quantile(f[...,2][fn],[0,.25,.5,.75,1]).tolist() if fn.any() else [],positive_vote_ratio_on_fn=float(f[...,4][fn].mean()) if fn.any() else None,diagnostic_not_causal_proof=True,teacher_witness_is_not_truth=True,target_rgb_used=False)
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,result['fn_partition'],flush=True)
if __name__=='__main__':main()
