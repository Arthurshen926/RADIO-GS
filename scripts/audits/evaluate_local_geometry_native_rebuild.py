"""Re-lift identical source masks/features onto new geometry; native AP control.
No old row-indexed field, memberships, seeds, or learned extents reused on new map.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.training.region_visibility import normalize_visible_region_mass
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks,binary_iou
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError(a.output)
 base=Path('/mnt/pool/sqy/results/RADIO-GS/output');old=base/'preserved_source_inventory_20260916'/a.scene;new=base/'local_calibrated_geometry_20260916'/a.scene;ar=new/'native_diagnostic_authority';auth=json.loads((ar/'responsibility.json').read_text())
 info=json.loads((old/'teachers.json').read_text());ep=old/'episodes.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];data=torch.load(ep,weights_only=False)
 assert data['teacher_manifest_sha256']==sha256_file(old/'teachers.json')
 sr=info['teacher_files']['siglip'];assert sha256_file(old/sr['path'])==sr['sha256'];teacher=torch.load(old/sr['path'],weights_only=False)
 fp=old/'exact_render_query_threshold96/field.pt';assert sha256_file(fp)==json.loads((fp.parent/'report.json').read_text())['field_sha256'];payload=torch.load(fp,weights_only=False)
 field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
 gr=json.loads((base/'geometry_only_repair_final_20260914'/a.scene/'report.json').read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);g=gr['results']['independent_scale'];parent=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device);parent.assert_binding(data['geometry_binding']);parent.assert_binding(payload['geometry_binding'])
 candidate=FrozenGaussianGeometry.load(new/'geometry.pt',expected_sha256=auth['metadata']['frozen_geometry_sha256'],device=a.device);candidate.assert_binding(auth['metadata']['frozen_geometry_binding'])
 masks=data['masks'].to(a.device).reshape(len(data['masks']),-1);views=data['proposal_views'];n=len(candidate.xyz);prob=torch.zeros(len(masks),n,device=a.device);seen=torch.zeros(n,dtype=torch.bool,device=a.device)
 assert auth['frame_indices']==info['train_frames']
 for index,record in enumerate(auth['views']):
  f=info['train_frames'][index];v=info['frames'].index(f);path=ar/record['relative_path'];assert sha256_file(path)==record['sha256'];hit=torch.load(path,map_location=a.device,weights_only=False)
  pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();w=hit['base_weights'].float();mass=torch.zeros(96*128,device=a.device).scatter_add_(0,pid,w)
  mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(96*128,n)).coalesce();rowsum=torch.sparse.sum(mat,dim=0).to_dense();selected=torch.where(views==v)[0].to(a.device)
  lifted,visible=normalize_visible_region_mass(torch.sparse.mm(mat.T,masks[selected].float().T),rowsum);prob[selected]=lifted.T.half().float();seen|=visible
 train=torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in views],device=a.device);descriptors=F.normalize(teacher['descriptors'].float().to(a.device),dim=1)
 with torch.no_grad():
  projected=descriptors@field.native_decoder.weight;mass=prob[train].sum(0);latent=prob[train].T@projected[train]/mass[:,None].clamp_min(1e-8);valid=seen&(mass>1e-8)&(latent.norm(dim=1)>1e-8)
  # Verify inherited basis and source fusion reconstruct the parent native prefix.
  pw=data['region_probability'][train.cpu()].float().to(a.device);pl=pw.T@projected[train]/pw.sum(0)[:,None].clamp_min(1e-8)
  torch.testing.assert_close(pl[field.observed],field.latent[field.observed,:64],atol=2e-5,rtol=2e-4)
  del pw,prob,masks,teacher,data
  rebuild_path=new/'native_rebuilt_diagnostic.pt';assert not rebuild_path.exists()
  torch.save(dict(latent=latent.cpu(),valid=valid.cpu(),geometry_binding=candidate.fingerprint(),authority_sha256=sha256_file(ar/'responsibility.json'),source_episode_sha256=sha256_file(ep),native_basis=field.native_decoder.weight.cpu(),diagnostic_only=True),rebuild_path)
  labels=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label');annotations,categories,sh,sw=load_labels(labels,a.scene);assert not set(annotations)&set(info['train_frames'])
  encoder=NativeThreeQueryEncoder(payload['query_encoder'],a.device);q=encoder.encode_text(categories).half().float();threshold=field.native_query_thresholds['text'](q);scale=field.readout_log_scale['text'].clamp(-4,6).exp()
  # Decoder is the original orthonormal source basis; cosine denominator stays in native space.
  decoded=latent@field.native_decoder.weight.T
  new_cos=F.normalize(decoded,dim=-1)@F.normalize(q,dim=-1).T;del decoded
  responses={'parent_native':(parent,field.observed,torch.sigmoid(scale*(field.native_cosine(q)-threshold))), 'new_geometry_native':(candidate,valid,torch.sigmoid(scale*(new_cos-threshold)))}
  rows={name:[] for name in responses};height=round(sh*480/sw)
  for frame,objects in sorted(annotations.items()):
   cam=prior.cameras[frame];k=cam.intrinsic.to(a.device).clone();k[0]*=480/cam.width;k[1]*=height/cam.height;targets=ground_truth_masks(objects,categories,height,480,sh,sw)
   for name,(model,observed,response) in responses.items():
    colors=torch.cat([response*observed[:,None],observed.float()[:,None]],1)
    image,alpha,_=rasterization(means=model.xyz,quats=model.quaternions,scales=model.scales,opacities=model.opacities.flatten(),colors=colors,viewmats=cam.world_to_camera.to(a.device)[None],Ks=k[None],width=480,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
    support=image[0,...,-1];known=(support/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);score=image[0,...,:len(q)]/support[...,None].clamp_min(1e-8)
    for ci,c in enumerate(categories):
     truth=targets[c].to(a.device)
     if not truth.any():continue
     value=torch.where(known,score[...,ci],torch.zeros_like(support));iou,_,_=binary_iou((value>.5)&known,truth);rows[name].append(dict(frame=frame,category=c,iou=iou,**binary_ranking_metrics(value,truth)))
  macro={name:{k:sum(r[k] for r in rr)/len(rr) for k in ['iou','average_precision']} for name,rr in rows.items()}
 result=dict(schema='local_geometry_native_rebuild.v1',scene=a.scene,diagnostic_only=True,complete_method=False,target_rgb_used=False,benchmark_masks_for_metrics_only=True,shared_source_masks_descriptors_and_native_basis=True,calibration='same inherited source query threshold; not refitted; AP primary',macro=macro,rows=rows,parent_field_sha256=sha256_file(fp),new_geometry_sha256=sha256_file(new/'geometry.pt'),new_authority_sha256=sha256_file(ar/'responsibility.json'),new_rebuilt_features_sha256=sha256_file(rebuild_path),source_episode_sha256=sha256_file(ep),source_teacher_sha256=sr['sha256'],implementation_sha256=sha256_file(__file__),native_elements=int(valid.sum()))
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,macro,flush=True)

if __name__=='__main__':main()
