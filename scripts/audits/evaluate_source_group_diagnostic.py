"""Separate candidate range ceilings from frozen semantic candidate ranking.

Oracle selections use evaluation labels only for diagnosis, never exported as
queries, supervision, candidate geometry, descriptors or production predictions.
"""
import argparse,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser()
 for key in ('root','geometry-report','candidates','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--scene',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--width',type=int,default=480);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError(a.output)
 def deny(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and any(s in str(args[0]) for s in ('episodes.pt','teacher_radio.pt','teacher_dino.pt','teacher_sam.pt','teacher_siglip.pt','grounded_text_supervision')):raise RuntimeError('training assets forbidden at diagnostic evaluation')
 sys.addaudithook(deny)
 receipt=json.loads(a.candidates.with_suffix('.json').read_text());assert sha256_file(a.candidates)==receipt['sha256']
 pack=torch.load(a.candidates,weights_only=False);assert pack['diagnostic_only'] and pack['source_only']
 info=json.loads((a.root/'teachers.json').read_text());assert pack['train_frames']==info['train_frames']
 auth=json.loads((a.root/'responsibility.json').read_text());assert sha256_file(a.root/'responsibility.json')==info['authority_sha256']
 assert sha256_file(a.geometry_report)==auth['metadata']['frozen_geometry_report_sha256']
 gr=json.loads(a.geometry_report.read_text());g=gr['results']['independent_scale'];geometry=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device);geometry.assert_binding(pack['geometry_binding'])
 identity=json.loads((a.root/'native_query_encoder.json').read_text());assert identity==pack['query_encoder'];encoder=NativeThreeQueryEncoder(identity,a.device)
 prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
 labelsroot=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label');annotations,categories,sh,sw=load_labels(labelsroot,a.scene)
 assert not set(annotations)&set(info['train_frames'])
 queries=F.normalize(encoder.encode_text(categories).half().float(),dim=-1)
 ranks={name:(desc.to(a.device)@queries.T).argsort(dim=0,descending=True) for name,desc in pack['descriptors'].items()}
 observed=pack['observed'].to(a.device);members=pack['memberships'].to(a.device);h=round(sh*a.width/sw);rows=[];previews=[]
 with torch.no_grad():
  for frame,objects in sorted(annotations.items()):
   camera=prior.cameras[frame];k=camera.intrinsic.to(a.device).clone();k[0]*=a.width/camera.width;k[1]*=h/camera.height
   masks=[]
   for start in range(0,len(members),32):
    colors=torch.cat([members[start:start+32].T.float(),observed[:,None].float()],1)
    image,alpha,_=rasterization(means=geometry.xyz,quats=geometry.get_rotation(),scales=geometry.get_scaling(),opacities=geometry.get_opacity().reshape(-1),colors=colors,viewmats=camera.world_to_camera.to(a.device)[None],Ks=k[None],width=a.width,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
    mass=image[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
    masks.append(((image[0,...,:-1]/mass[...,None].clamp_min(1e-8)>.5)&valid[...,None]).permute(2,0,1).flatten(1))
   masks=torch.cat(masks);targets=ground_truth_masks(objects,categories,h,a.width,sh,sw)
   for ci,c in enumerate(categories):
    target=targets[c].to(a.device).flatten()
    if not target.any():continue
    intersection=(masks&target).sum(1);union=(masks|target).sum(1);ious=intersection/union.clamp_min(1)
    best=int(ious.argmax());row={'frame':frame,'category':c,'oracle_single_group_iou':float(ious[best]),'oracle_group':best}
    for name,rank in ranks.items():
     ordered=rank[:,ci];row[name]={'top1_iou':float(ious[ordered[0]]),'oracle_rank':int(torch.where(ordered==best)[0][0])+1,'top5_oracle_iou':float(ious[ordered[:5]].max())}
    rows.append(row)
    if frame==min(annotations):previews.append((c,target.cpu().reshape(h,a.width),masks[best].cpu().reshape(h,a.width),masks[ranks['field_mean'][0,ci]].cpu().reshape(h,a.width),masks[ranks['native_region'][0,ci]].cpu().reshape(h,a.width)))
   print(a.scene,frame,len(rows),flush=True)
 output={'schema':'source_group_failure_localization.v1','candidate_sha256':receipt['sha256'],'implementation_sha256':sha256_file(__file__),'width':a.width,'prediction_threshold':.5,'diagnostic_only':True,'oracle_uses_ground_truth':True,'selection_without_ground_truth':'top1 native cosine; all source groups, no target-view filtering','cohort':'full annotated text development; category-union targets','rows':rows,'macro':{'oracle_single_group_iou':sum(x['oracle_single_group_iou'] for x in rows)/len(rows)}}
 for name in ranks:output['macro'][name]={key:sum(x[name][key] for x in rows)/len(rows) for key in ('top1_iou','oracle_rank','top5_oracle_iou')}
 a.output.write_text(json.dumps(output,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(len(previews),4,figsize=(10,2*len(previews)),squeeze=False)
 for i,(category,*pictures) in enumerate(previews):
  for j,(picture,title) in enumerate(zip(pictures,['GT','best range (oracle)','field mean top1','native region top1'])):
   axes[i,j].imshow(picture,cmap='gray',vmin=0,vmax=1);axes[i,j].set_title(category+' / '+title,fontsize=8);axes[i,j].axis('off')
 fig.tight_layout();fig.savefig(a.output.with_suffix('.png'),dpi=100);plt.close(fig);print(json.dumps(output['macro']),flush=True)
if __name__=='__main__':main()
