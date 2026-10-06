"""NVOS public positive-input evaluation: positive scribble -> SAM crop -> image field.

No target RGB may be opened. Prediction is persisted before target GT is read.
Only the positive scribble is authorized. The all-view public contract allows
more RGB access than this frozen-map readout needs; that access is not used.
"""
import argparse,json,sys,os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',default='fern');p.add_argument('--signed-refinement',action='store_true');p.add_argument('--signed-bound-control',type=float,choices=[8.]);p.add_argument('--signed-policy',choices=['fixed','adaptive','minimal'],default='fixed');p.add_argument('--query-root',type=Path,default=Path('/mnt/pool/sqy/results/RADIO-GS/output/nvos_reference_sam_queries_20260915'));p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry-report',type=Path,required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.signed_refinement or a.signed_bound_control is not None or a.signed_policy!='fixed':raise ValueError('Positive-only adapter does not authorize signed refinement')
 assert not a.output.exists();a.output.mkdir(parents=True)
 mp=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json');manifest=json.loads(mp.read_text());scene=next(s for s in manifest['scenes'] if s['scene_id']==a.scene)
 target=next(f for f in scene['frames'] if f['frame_id'] in scene['evaluation_frame_ids']);reference=next(f for f in scene['frames'] if f['frame_id']==scene['prompt']['frame_id'])
 forbidden={str(Path(target[k]).resolve()) for k in ('rgb_path','annotation_rgb_path')}
 gtpath=Path(target['gt_mask_path']);prediction_saved=False
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   name=str(Path(os.fsdecode(args[0])).resolve())
   if name in forbidden or name==str(Path(scene['prompt']['negative_path']).resolve()):raise RuntimeError('NVOS target RGB and negative prompt forbidden')
   if name==str(gtpath.resolve()) and not prediction_saved:raise RuntimeError('target GT forbidden before prediction is saved')
   if any(x in name for x in ('teacher_dino.pt','teacher_siglip.pt','teacher_sam.pt','teacher_radio.pt','episodes.pt','queries.pt','caption_region_evidence.pt','source_grounded_text.pt','grounded_text_supervision.pt')):raise RuntimeError('training feature or supervision access forbidden')
 sys.addaudithook(guard)
 source=json.loads((a.root/'source_frames.json').read_text());assert all(Path(f['source_file']).stem not in scene['evaluation_frame_ids'] for f in source['frames'])
 report=json.loads(a.geometry_report.read_text());record=report['results']['independent_scale'];geo=FrozenGaussianGeometry.load(record['geometry'],expected_sha256=record['geometry_sha256'],device=a.device)
 prior=MogeGeometryPrior(report['source_report'],report['source_report_sha256'])
 assert int(target['canonical_index']) not in report['mapping_frames']
 upstream={r['role']:r for r in prior.report['geometry_receipt']['inputs']}
 authority=json.loads(Path(upstream['source_authority']['path']).read_text())
 names=authority['camera_name_to_frame_index']
 for frame in (reference,target):
  indices=[int(index) for name,index in names.items() if Path(name).stem==frame['frame_id']]
  assert indices==[int(frame['canonical_index'])], 'camera filename mapping differs'
 assert int(target['canonical_index']) not in authority['frame_indices']
 query_root=a.query_root/a.scene
 qr=json.loads((query_root/'receipt.json').read_text());assert qr['negative_scribble_opened'] is False and qr['input_contract']=='nvos-ludvig-online-all-view-eight-v1';assert qr['reference_frame']==reference['frame_id'] and qr['target_reads_forbidden']
 assert Path(qr['reference_rgb']['path']).resolve()==Path(reference['rgb_path']).resolve()
 assert sha256_file(reference['rgb_path'])==qr['reference_rgb']['sha256']
 for sign in ('positive',):assert qr['official_scribbles'][sign]['sha256']==manifest['protocol']['prompt_asset_sha256'][a.scene][sign]
 crop=qr['image_query_crop'];assert sha256_file(crop['path'])==crop['sha256']
 identity=json.loads((a.root/'native_query_encoder.json').read_text());encoder=NativeThreeQueryEncoder(identity,a.device);q=encoder.encode_image(Image.open(crop['path']).convert('RGB'))
 encoder.vision=None
 import gc
 gc.collect();torch.cuda.empty_cache()
 width,height=qr['prompt_raster_wh'];predictions={};supports={};bindings={};reference_constraints={};refinements={}
 with torch.no_grad():
  for run in a.runs:
   receipt=json.loads((a.root/run/'report.json').read_text());field,payload=load_element_retrieval_field(a.root/run/'field.pt',expected_sha256=receipt['field_sha256'],geometry=geo,expected_query_encoder=identity,device=a.device)
   bindings[run]=receipt['field_sha256'];scores=field.retrieve(q,'image')['scores']
   for role,frame in [('reference',reference),('target',target)]:
    camera=prior.cameras[int(frame['canonical_index'])]
    k=camera.intrinsic.to(a.device).clone();k[0]*=width/camera.width;k[1]*=height/camera.height
    colors=torch.cat([scores,field.observed_for('image').float()[:,None]],-1)
    rendered,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,viewmats=camera.world_to_camera.to(a.device)[None],Ks=k[None],width=width,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
    mass=rendered[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
    response=torch.where(valid,rendered[0,...,0]/mass.clamp_min(1e-8),0).cpu();binary=response>=.5
    if role=='target':predictions[run]=response;supports[run]=valid.cpu();torch.save(dict(response=response,supported=valid.cpu(),threshold=.5,field_sha256=receipt['field_sha256']),a.output/(run+'.pt'));Image.fromarray(binary.numpy().astype(np.uint8)*255).save(a.output/(run+'.png'))
    else:
     result={}
     for sign in ('positive',):
      item=qr['official_scribbles'][sign];assert sha256_file(item['path'])==item['sha256'];mask=np.asarray(Image.open(item['path']));mask=np.any(mask>127,axis=2) if mask.ndim==3 else mask>127
      assert mask.shape==(height,width)
      result[sign+'_agreement']=float((binary if sign=='positive' else ~binary)[torch.from_numpy(mask)].float().mean())
     reference_constraints[run]=result
 prediction_saved=True
 assert sha256_file(gtpath)==target['ground_truth_sha256'];gt=np.asarray(Image.open(gtpath));gt=np.any(gt>127,axis=2) if gt.ndim==3 else gt>127
 # Public all-view comparator task uses continuous bilinear resize before threshold.
 # The strict-unseen manifest supplies assets/cameras only; its nearest rule is not this task.
 truth=torch.from_numpy(gt);results={}
 for run,response in predictions.items():
  response=torch.from_numpy(cv2.resize(response.numpy(),(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR))
  supports[run]=torch.from_numpy(cv2.resize(supports[run].numpy().astype(np.uint8),(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool))
  binary=response>=.5;results[run]=dict(iou=float((binary&truth).sum()/(binary|truth).sum()) if (binary|truth).any() else float(manifest['protocol']['empty_union_value']),pixel_accuracy=float((binary==truth).float().mean()),**binary_ranking_metrics(response.flatten(),truth.flatten()),positive_support_recall=float(supports[run][truth].float().mean()),supported_pixel_fraction=float(supports[run].float().mean()),reference_signed_agreement=reference_constraints[run],prediction_sha256=sha256_file(a.output/(run+'.pt')))
 result=dict(schema='nvos_positive_field_evaluation.v1',contract='nvos-ludvig-online-all-view-eight-v1',negative_scribble_opened=False,scene=a.scene,manifest_sha256=sha256_file(mp),geometry_report_sha256=sha256_file(a.geometry_report),geometry_sha256=record['geometry_sha256'],query_receipt_sha256=sha256_file(query_root/'receipt.json'),query_root=str(a.query_root),sam_reference_complex_rope_preserved=qr.get('complex_rope_buffers_preserved',False),fields=bindings,target_frame=target['frame_id'],target_rgb_opened=False,target_gt_used_after_prediction_only=True,training_feature_reads_forbidden=True,complete_joint_method=False,official_signed_constraints_guaranteed=False,signed_refinement=a.signed_refinement,signed_policy=a.signed_policy,signed_bound_control=a.signed_bound_control,refinement_records=refinements,threshold=.5,threshold_comparison='greater_or_equal',score_semantics='observed-alpha normalized image membership probability; not legacy foreground/background cosine margin',score_resize='cv2.INTER_LINEAR continuous score to native GT grid before threshold; public all-view task, manifest asset identities retained',grid_wh=[width,height],scoring_grid_wh=[gt.shape[1],gt.shape[0]],results=results,
 implementation_sources={f:sha256_file(f) for f in [__file__,'radio_gs/models/element_retrieval_field.py','radio_gs/models/bounded_native_residual.py']})
 (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(results,flush=True)


if __name__=='__main__':main()
