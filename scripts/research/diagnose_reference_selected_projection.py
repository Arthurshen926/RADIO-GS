"""Diagnostic only: actual subset silhouettes, not the frozen NVOS response metric."""
import argparse,json
from pathlib import Path
import numpy as np,torch,cv2
from PIL import Image
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='output/reference_integration');p.add_argument('--field-template',default='subset/{scene}/sets/{scene}/typed_readout.pt');a=p.parse_args()
    torch.set_num_threads(4);root=Path(a.root);rows=[]
    for scene in ['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex']:
        config=json.loads(Path(f'configs/consolidated/nvos/{scene}.scene.json').read_text());evaluation=json.loads(Path(f'configs/consolidated/nvos/{scene}.evaluation.json').read_text());data=torch.load(root/a.field_template.format(scene=scene),map_location='cpu',weights_only=False)
        binding=data['selection_geometry'];geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda');gp=MogeGeometryPrior(config['camera_report'],config['camera_report_sha256']);cam=gp.cameras[evaluation['target_frame']];width,height=evaluation['raster_wh'];k=cam.intrinsic.cuda().clone();k[0]*=width/cam.width;k[1]*=height/cam.height
        selected=(data['selection_scores'][:,0]>.5)&data['selection_observed'];selected=selected.cuda()
        _,alpha,_=rasterization(means=geo.xyz[selected],quats=geo.quaternions[selected],scales=geo.scales[selected],opacities=geo.opacities.flatten()[selected],colors=torch.ones(int(selected.sum()),1,device='cuda'),viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=width,height=height,near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False)
        truth=np.asarray(Image.open(evaluation['gt_mask']));truth=np.any(truth>127,axis=2) if truth.ndim==3 else truth>127
        response=cv2.resize(alpha[0,...,0].cpu().numpy(),(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR);prediction=response>10.5/255
        rows.append(dict(scene=scene,selected_projection_iou=float((prediction&truth).sum()/max(1,(prediction|truth).sum())),selected_elements=int(selected.sum())))
    result=dict(official_nvos_score=False,protocol='diagnostic subset alpha >10.5/255 after bilinear resize; NOT continuous normalized NVOS response',scenes=rows,macro_iou=float(np.mean([r['selected_projection_iou'] for r in rows])))
    (root/'subset_projection_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
