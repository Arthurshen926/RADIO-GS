"""Matched fixed-mask diagnostics: visibility, alpha cutoff, hard 3D selection.
No fitting or threshold search. The .5 alpha arm is diagnostic, not promotion.
"""
import argparse,json
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks

def main():
 p=argparse.ArgumentParser();p.add_argument('--selected-root',type=Path,required=True);p.add_argument('--field',type=Path,required=True);p.add_argument('--geometry-report',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
 r=json.loads((a.selected_root/'report.json').read_text());protocol=r['protocol'];assert sha256_file(a.selected_root/'selection.pt')==r['selection_sha256'];assert sha256_file(a.field)==protocol['field_sha256'];payload=torch.load(a.field,map_location='cpu',weights_only=False);observed=payload['state_dict']['observed'].to(a.device);del payload
 sel=torch.load(a.selected_root/'selection.pt',weights_only=False);selected=sel['selected'].to(a.device);gr=json.loads(a.geometry_report.read_text());g=gr['results']['independent_scale'];assert g['geometry_sha256']==protocol['geometry_sha256'];geometry=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
 labelroot=Path(protocol['labels'][0]['path']).parent.parent
 for item in protocol['labels']:assert sha256_file(item['path'])==item['sha256']
 annotations,categories,h,w=load_labels(labelroot,protocol['scene']);assert categories==sel['categories'];rows=[]
 def raster(ids,colors,camera,k):
  return rasterization(means=geometry.get_xyz()[ids],quats=geometry.get_rotation()[ids],scales=geometry.get_scaling()[ids],opacities=geometry.get_opacity().reshape(-1)[ids],colors=colors,viewmats=camera.world_to_camera.to(a.device)[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
 def metrics(pred,target):
  tp=int((pred&target).sum());fp=int((pred&~target).sum());fn=int((~pred&target).sum());return dict(iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),false_positive_pixels=fp,false_negative_pixels=fn)
 with torch.no_grad():
  for frame,objects in sorted(annotations.items()):
   camera=prior.cameras[frame];k=camera.intrinsic.to(a.device).clone();k[0]*=w/camera.width;k[1]*=h/camera.height
   image,alpha,_=raster(torch.ones(len(observed),dtype=torch.bool,device=a.device),torch.cat([selected.float(),observed.float()[:,None]],1),camera,k);mass=image[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);hard=(image[0,:,:,:-1]/mass[...,None].clamp_min(1e-8)>.5)&valid[...,None]
   targets=ground_truth_masks(objects,categories,h,w,h,w)
   for ci,category in enumerate(categories):
    target=targets[category].to(a.device)
    if not target.any():continue
    ids=selected[:,ci]
    if ids.any():_,subsetalpha,_=raster(ids,torch.ones(int(ids.sum()),1,device=a.device),camera,k);subsetalpha=subsetalpha[0,...,0]
    else:subsetalpha=torch.zeros(h,w,device=a.device)
    public=(subsetalpha*255+.5).floor().clamp(0,255)>10
    saved=torch.from_numpy(np.array(Image.open(a.selected_root/f'frame_{frame:05d}_{category}.png'))).to(a.device)>10;assert torch.equal(public,saved)
    rows.append(dict(frame=frame,category=category,full_scene_hard=metrics(hard[...,ci],target),selected_alpha_half=metrics(subsetalpha>.5,target),selected_public_cutoff=metrics(public,target)))
 names=['full_scene_hard','selected_alpha_half','selected_public_cutoff'];report=dict(schema='matched_selected_render_diagnostic.v1',training=False,threshold_search=False,selected_report_sha256=sha256_file(a.selected_root/'report.json'),implementation_sha256=sha256_file(__file__),notes='Same fixed element selection and full-resolution polygon target. Half-alpha arm is a fixed diagnostic, not a selected or public evaluation policy. Differences are not additive causal error attribution.',observations=rows,macro={name:{metric:sum(row[name][metric] for row in rows)/len(rows) for metric in ['iou','precision','recall']} for name in names});a.output.write_text(json.dumps(report,indent=2)+'\n');print(report['macro'])
if __name__=='__main__':main()
