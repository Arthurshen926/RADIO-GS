"""Fixed reference-index qualitative checks; target RGB never used for inference."""
import argparse,json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',default='cuda:4');a=p.parse_args();torch.set_num_threads(4)
 base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=base/'preserved_source_inventory_20260916'/a.scene;result=json.loads(Path(f'paper/artifacts/{a.scene}_scale_affinity_reference_20260916.json').read_text());refs=result['references'];chosen=sorted({0,len(refs)//3,2*len(refs)//3,len(refs)-1})
 gr=json.loads((base/'geometry_only_repair_final_20260914'/a.scene/'report.json').read_text());rec=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(rec['geometry'],expected_sha256=rec['geometry_sha256'],device=a.device);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);identity=json.loads((root/'native_query_encoder.json').read_text());encoder=NativeThreeQueryEncoder(identity,a.device)
 names=['preserved96_protected_imagecal_balanced','scale_affinity_v1_direct_control'];fields=[]
 for name in names:
  path=root/name/'field.pt';field,payload=load_element_retrieval_field(path,expected_sha256=sha256_file(path),geometry=geo,expected_query_encoder=identity,device=a.device);fields.append(field)
 assert torch.equal(fields[0].observed,fields[1].observed)
 normal=payload['query_context']['prompt_world_normalization'];qh,qw=payload['query_context']['prompt_raster_size'];ann,categories,sh,sw=load_labels(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label'),a.scene);width=480;height=round(sh*width/sw);pictures=[]
 with torch.no_grad():
  for index in chosen:
   ref=refs[index];assert sha256_file(Path(ref['image_path']))==ref['image_sha256'];q=encoder.encode_prompt(ref['image_path'],ref['pixels'],geo.xyz[ref['seedrows']],normal['minimum'],normal['extent'],qh,qw).half().float();obs=next(o for o in result['runs'][names[0]]['observations'] if o['reference']==index);frame=obs['target_frame'];camera=prior.cameras[frame];k=camera.intrinsic.to(a.device).clone();k[0]*=width/camera.width;k[1]*=height/camera.height
   colors=torch.cat([f.retrieve(q,'prompt')['scores'] for f in fields]+[fields[0].observed.float()[:,None]],dim=1)
   image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities[:,0],colors=colors,viewmats=camera.world_to_camera.to(a.device)[None],Ks=k[None],width=width,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
   mass=image[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);response=image[0,...,:2]/mass[...,None].clamp_min(1e-8);pred=(response>.5)&valid[...,None];gt=ground_truth_masks(ann[frame],[ref['category']],height,width,sh,sw)[ref['category']]
   measured=[float(((pred[...,i].cpu())&gt).sum()/((pred[...,i].cpu())|gt).sum().clamp_min(1)) for i in range(2)]
   for i,name in enumerate(names):
    expected=next(o for o in result['runs'][name]['observations'] if o['reference']==index and o['target_frame']==frame)['prompt']['iou'];assert abs(measured[i]-expected)<1e-6
   pictures.append((ref['category'],frame,gt,pred.cpu(),measured))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(len(pictures),3,figsize=(10,2.7*len(pictures)),squeeze=False)
 for i,(name,frame,gt,pred,metrics) in enumerate(pictures):
  for j,(picture,label) in enumerate([(gt,'GT'),(pred[...,0],f'learned query: {metrics[0]*100:.1f}%'),(pred[...,1],f'registered structure: {metrics[1]*100:.1f}%')]):
   axes[i,j].imshow(picture,cmap='gray',vmin=0,vmax=1);axes[i,j].set_title(f'{name} / frame {frame}\n{label}',fontsize=10);axes[i,j].axis('off')
 fig.suptitle(a.scene+' / fixed reference indices; first cross-view target; IoU');fig.tight_layout();path=Path(f'paper/artifacts/{a.scene}_registered_structure_comparison_20260916.png');fig.savefig(path,dpi=120);plt.close(fig);print(path,flush=True)
if __name__=='__main__':main()
