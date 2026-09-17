"""Source-only 2D geometry render and static 3D plot for the new frozen bundle."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--geometry-report',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--label',default='Frozen source geometry');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
gr=json.loads(a.geometry_report.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device)
f=gr['heldout_frames'][len(gr['heldout_frames'])//2];c=prior.cameras[f];rec=prior.records[f];rgbpath=Path(rec['source_image']);assert sha256_file(rgbpath)==rec['source_image_sha256'];rgb=Image.open(rgbpath).convert('RGB')
w=640;h=round(c.height*w/c.width);k=c.intrinsic.to(a.device).clone();k[0]*=w/c.width;k[1]*=h/c.height;pose=c.world_to_camera.to(a.device)
with torch.no_grad():
 z=(geo.xyz@pose[2,:3]+pose[2,3]);colors=torch.cat([z[:,None],(geo.normals@pose[:3,:3].T+1)/2],1)
 im,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,viewmats=pose[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
 mass=alpha[0,...,0];image=(im[0]/mass[...,None].clamp_min(1e-8)).cpu().numpy();valid=(mass>.5).cpu().numpy();depth=np.where(valid,image[...,0],np.nan);normal=np.where(valid[...,None],np.clip(image[...,1:],0,1),0)
fig=plt.figure(figsize=(15,8));axes=[fig.add_subplot(2,2,i+1) for i in range(3)]
axes[0].imshow(rgb.resize((w,h)));axes[0].set_title(f'Authorized source view {f} (excluded from fusion)')
lo,hi=np.nanpercentile(depth,[1,99]);obj=axes[1].imshow(depth,cmap='viridis',vmin=lo,vmax=hi);axes[1].set_title('Rendered depth, COLMAP units (1–99% color limits)');fig.colorbar(obj,ax=axes[1],shrink=.7)
axes[2].imshow(normal);axes[2].set_title('Rendered camera-space normals; alpha > 0.5')
for ax in axes:ax.axis('off')
ax=fig.add_subplot(2,2,4,projection='3d');stride=max(1,len(geo.xyz)//30000);xyz=geo.xyz[::stride].cpu().numpy();color=(geo.normals[::stride].cpu().numpy()+1)/2;ax.scatter(*xyz.T,c=color,s=.25,rasterized=True)
centres=torch.stack([prior.cameras[v].camera_to_world[:3,3] for v in gr['mapping_frames']]).numpy();ax.plot(*centres.T,color='black',linewidth=.7);ax.set_title('Frozen surface carriers and mapping camera trajectory');ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z');ax.set_box_aspect(np.ptp(xyz,axis=0).clip(1e-4))
fig.suptitle(a.label+'; no target RGB / GT',fontsize=13);fig.tight_layout();fig.savefig(a.output,dpi=150);plt.close(fig)
a.output.with_suffix('.json').write_text(json.dumps(dict(geometry_report_sha256=sha256_file(a.geometry_report),geometry_sha256=g['geometry_sha256'],source_frame=f,source_image_sha256=rec['source_image_sha256'],target_rgb_or_gt_opened=False,source_view_excluded_from_fusion=True,plot_sha256=sha256_file(a.output),implementation_sha256=sha256_file(__file__)),indent=2)+'\n')
print(a.output)
