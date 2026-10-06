"""Source-only diagnostic gallery: no target RGB or target labels."""
import json
from pathlib import Path
import torch,numpy as np
from PIL import Image
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior

@torch.inference_mode()
def main():
    torch.set_num_threads(4);out=Path('output/source_incompatibility_gallery');out.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(6,4,figsize=(16,18));records=[];row=0
    for name in ['trex','orchids']:
        scene=json.loads(Path(f'configs/consolidated/nvos/{name}.scene.json').read_text());query=json.loads(Path(f'configs/consolidated/nvos/{name}.query.json').read_text());diag=json.loads(Path(f'output/reference_representability/{name}/diagnosis.json').read_text());rank=sorted([r for r in diag['rows'] if r['training_domain']=='full'],key=lambda r:r['metrics']['single480']['full']['iou']-r['metrics']['joint480']['full']['iou'],reverse=True);frames=[query['reference_frame']]+[r['frame'] for r in rank if r['frame']!=query['reference_frame']][:2]
        field=torch.load(f'output/reference_representability/{name}/joint_full.pt',map_location='cpu',weights_only=False);tracks=torch.load(f'output/reference_integration/fresh_tracks/{name}/tracks.pt',map_location='cpu',weights_only=False);b=scene['geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda');gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);h,w=192,256
        for frame in frames:
            cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
            rendered,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=field['elements'].cuda(),viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False);p=(rendered[0,...,0]/alpha[0,...,0].clamp_min(1e-8)).cpu().numpy();t=F.interpolate(tracks['frames'][frame]['logits'].float()[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0].sigmoid().numpy();rgb=np.array(Image.open(scene['source_rgb'][str(frame)]).convert('RGB').resize((w,h)));known=alpha[0,...,0].cpu().numpy()>1e-8;diff=np.where(known,p-t,0)
            for ax,im,title in zip(axes[row],[rgb,t,p,diff],[f'{name} source {frame}','source SAM probability','joint rendered response','response minus source']):
                ax.imshow(im,cmap='coolwarm' if title.startswith('response minus') else 'viridis',vmin=-1 if title.startswith('response minus') else 0,vmax=1) if im.ndim==2 else ax.imshow(im);ax.set_title(title);ax.axis('off')
            records.append(dict(scene=name,frame=frame,reference=frame==query['reference_frame'],source_rgb=scene['source_rgb'][str(frame)]));row+=1
    fig.tight_layout();fig.savefig(out/'source_conflicts.jpg',dpi=110);(out/'gallery.json').write_text(json.dumps(dict(records=records,selection='reference plus two largest single/joint source fit gaps',target_labels_used=False),indent=2)+'\n')
if __name__=='__main__':main()
