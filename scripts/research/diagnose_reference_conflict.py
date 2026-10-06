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
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--scene-input');p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4)
    scene=json.loads(Path(a.scene_input or f'configs/consolidated/nvos/{a.scene}.scene.json').read_text());ev=json.loads(Path(f'configs/consolidated/nvos/{a.scene}.evaluation.json').read_text());root=Path('output/reference_integration');field=torch.load(root/'fresh'/a.scene/'field.pt',map_location='cpu',weights_only=False);tracks=torch.load(root/'fresh_tracks'/a.scene/'tracks.pt',map_location='cpu',weights_only=False)
    b=scene['geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda');gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);n=len(geo.xyz);h,w=scene['source_fit_hw'];radius=scene['boundary_radius'];stats=torch.zeros(n,4,device='cuda');rawstats=torch.zeros_like(stats)
    for frame in scene['source_frames']:
        rec=tracks['frames'].get(frame)
        if rec is None or rec['object_logit']<=0:continue
        cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
        hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['weights'].float();total=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight);weight=weight/total[pid].clamp_min(1e-8)
        y=F.interpolate(rec['logits'].float()[None,None].cuda(),size=(h,w),mode='bilinear',align_corners=False);binary=(y>0).float();pos=(-F.max_pool2d(-binary,2*radius+1,1,radius)>.5).flatten();neg=(F.max_pool2d(binary,2*radius+1,1,radius)<.5).flatten();raw=binary.flatten()
        pm=torch.zeros(n,device='cuda').scatter_add_(0,gid,weight*pos.float()[pid]);nm=torch.zeros(n,device='cuda').scatter_add_(0,gid,weight*neg.float()[pid]);m=pm+nm;p=pm/m.clamp_min(1e-8)
        rm=torch.zeros(n,device='cuda').scatter_add_(0,gid,weight);rp=torch.zeros(n,device='cuda').scatter_add_(0,gid,weight*raw[pid]);pr=rp/rm.clamp_min(1e-8)
        rawstats[:,0]+=rm;rawstats[:,1]+=rp;rawstats[:,2]+=rm*pr.square();rawstats[:,3]+=rm*pr*(1-pr)
        stats[:,0]+=m;stats[:,1]+=pm;stats[:,2]+=m*p.square();stats[:,3]+=m*p*(1-p)
    mean=stats[:,1]/stats[:,0].clamp_min(1e-8);within=stats[:,3]/stats[:,0].clamp_min(1e-8);between=(stats[:,2]/stats[:,0].clamp_min(1e-8)-mean.square()).clamp_min(0);variance=mean*(1-mean)
    rawmean=rawstats[:,1]/rawstats[:,0].clamp_min(1e-8);rawwithin=rawstats[:,3]/rawstats[:,0].clamp_min(1e-8);rawbetween=(rawstats[:,2]/rawstats[:,0].clamp_min(1e-8)-rawmean.square()).clamp_min(0)
    observed=field['observed'].cuda();values=field['elements'].cuda();colors=torch.cat([values,observed[:,None].float(),within[:,None],between[:,None],variance[:,None],rawwithin[:,None],rawbetween[:,None]],1)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.save(dict(within=within.cpu(),between=between.cpu(),mean=mean.cpu(),mass=stats[:,0].cpu(),geometry=b),out/'element_conflict.pt')
    w,h=ev['raster_wh'];cam=gp.cameras[ev['target_frame']];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
    image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False)
    mass=image[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);response=torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0);fractions=image[0,...,1:]/alpha[0].clamp_min(1e-8)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);arrays=dict(response=response.cpu().numpy(),alpha=alpha[0,...,0].cpu().numpy(),fractions=fractions.cpu().numpy());np.savez_compressed(out/'target_evidence.npz',**arrays)
    # All source statistics and target projections fixed before GT access.
    gt=np.array(Image.open(ev['gt_mask']));gt=np.any(gt>127,axis=2) if gt.ndim==3 else gt>127;size=(gt.shape[1],gt.shape[0]);r=cv2.resize(arrays['response'],size);opacity=cv2.resize(arrays['alpha'],size);f=cv2.resize(arrays['fractions'],size);fn=gt&(r<ev['threshold']);remaining=fn.copy();parts={}
    groups=dict(true_positive=gt&(r>=ev['threshold']),false_negative=fn,false_positive=(~gt)&(r>=ev['threshold']),true_negative=(~gt)&(r<ev['threshold']))
    summary={}
    for name,group in groups.items():
        wi=f[...,1][group];be=f[...,2][group];total=float((wi+be).sum())
        summary[name]=dict(pixels=int(group.sum()),within_mean=float(wi.mean()) if group.any() else 0,between_mean=float(be.mean()) if group.any() else 0,within_fraction_of_conflict=float(wi.sum()/max(total,1e-12)),between_fraction_of_conflict=float(be.sum()/max(total,1e-12)))
        rawtotal=float(f[...,4][group].sum()+f[...,5][group].sum());summary[name]['raw_between_fraction_of_conflict']=float(f[...,5][group].sum()/max(rawtotal,1e-12))
    result=dict(scene=a.scene,groups=summary,variance_identity_max_error=float((variance-within-between).abs().max()),diagnostic_not_causal_proof=True,teacher_witness_is_not_truth=True,target_rgb_used=False,interpretation='within-view footprint label mixing vs between-view mean-label variance; both may reflect geometry or teacher errors')
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,result['groups']['false_negative'],flush=True)
if __name__=='__main__':main()
