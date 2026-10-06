"""Frozen positive-only NVOS readout and scoring, separate from inference."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.training.retrieval_metrics import binary_ranking_metrics


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--field',required=True);p.add_argument('--subset-field',action='store_true');p.add_argument('--typed-readout',action='store_true');p.add_argument('--selected-subset',action='store_true',help='Supplementary actual extracted-set silhouette; never replaces the retained response metric');p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);scene=json.loads(Path(a.scene).read_text());evaluation=json.loads(Path(a.evaluation).read_text());d=torch.load(a.field,map_location='cpu',weights_only=False)
    if a.selected_subset:
        binding=d['selection_geometry'];values=d['selection_scores'];observed=d['selection_observed']
    elif a.typed_readout:
        binding=d['semantic_geometry'];values=d['scores'];observed=d['semantic_observed']
    else:
        binding=d['selection_geometry'] if a.subset_field else d['geometry'];values=d['selection_scores'] if a.subset_field else d['elements'];observed=d['selection_observed'] if a.subset_field else d['observed']
    values=values.cuda();observed=observed.cuda()
    geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda');gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);cam=gp.cameras[evaluation['target_frame']];width,height=evaluation['raster_wh'];k=cam.intrinsic.cuda().clone();k[0]*=width/cam.width;k[1]*=height/cam.height
    opacity=geo.opacities.flatten()
    if a.selected_subset:opacity=opacity*(values[:,0]>.5)*observed
    image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=opacity,colors=torch.cat([values,observed.float()[:,None]],1),viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=width,height=height,near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False)
    if a.selected_subset:response=alpha[0,...,0].cpu()
    else:
        mass=image[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);response=torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0).cpu()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.save(dict(response=response),out/'response.pt')
    # Prediction is finalized before annotation is opened.
    gt=np.asarray(Image.open(evaluation['gt_mask']));truth=torch.from_numpy(np.any(gt>127,axis=2) if gt.ndim==3 else gt>127)
    resized=torch.from_numpy(cv2.resize(response.numpy(),(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR))
    binary=((resized.clamp(0,1)*255+.5).floor()>10) if a.selected_subset else resized>=evaluation['threshold']
    union=(binary|truth).sum();metrics=dict(iou=float((binary&truth).sum()/union) if union else 1.,**binary_ranking_metrics(resized.flatten(),truth.flatten()))
    operator='selected-set alpha, 8-bit alpha > 10; supplementary projection diagnostic' if a.selected_subset else 'full-scene observed-support-normalized response'
    (out/'evaluation.json').write_text(json.dumps(dict(scene=scene['scene'],metrics=metrics,protocol=evaluation['protocol'],field=a.field,target_rgb_used=False,output_operator=operator,subset_trained=a.subset_field,typed_readout=a.typed_readout,selected_subset=a.selected_subset,development=True),indent=2)+'\n');print(scene['scene'],metrics,flush=True)


if __name__=='__main__':main()
