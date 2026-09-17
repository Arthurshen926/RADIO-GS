"""Real polygon-derived reference image/registered prompt cross-view queries.

A development protocol: first annotated reference per category supplies query
mask/points; other annotated views are targets. No target RGB is encoded.
"""
import argparse,json,gc,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.models.signed_query_refinement import condition_signed_query,retrieve_latent_query,SignedEvidenceUnavailable
from radio_gs.querying.native_image_conditioning import condition_native_image_query
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks,binary_iou
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry-report',type=Path,required=True);p.add_argument('--scene',required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--label-root',type=Path,default=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label'));p.add_argument('--device',default='cuda:4');p.add_argument('--refine-prompts',action='store_true');p.add_argument('--condition-images',action='store_true');p.add_argument('--minimal-prompt-refinement',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists() or a.output.with_suffix('.protocol.json').exists():raise FileExistsError(a.output)
 def deny_training(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and any(x in str(args[0]) for x in ('caption_region_evidence.pt','caption_grounded_regions.pt','caption_source_phrases.json','source_grids.pt','teacher_radio.pt','teacher_dino.pt','teacher_sam.pt','teacher_siglip.pt','episodes.pt','queries.pt','grounded_text_supervision','source_grounded_text','source_context_text','consistent_native_element_teacher')):raise RuntimeError('training evidence access forbidden')
 sys.addaudithook(deny_training)
 info=json.loads((a.root/'teachers.json').read_text());source=json.loads((a.root/'source_frames.json').read_text());auth=json.loads((a.root/'responsibility.json').read_text());identity=json.loads((a.root/'native_query_encoder.json').read_text())
 if sha256_file(a.root/'responsibility.json')!=info['authority_sha256'] or sha256_file(a.geometry_report)!=auth['metadata']['frozen_geometry_report_sha256']:raise ValueError('geometry authority differs')
 report=json.loads(a.geometry_report.read_text());g=report['results']['independent_scale'];prior=MogeGeometryPrior(report['source_report'],report['source_report_sha256'])
 geometry=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device)
 records={}
 for run in a.runs:
  r=json.loads((a.root/run/'report.json').read_text());path=a.root/run/'field.pt'
  if sha256_file(path)!=r['field_sha256']:raise ValueError('field differs')
  records[run]=r['field_sha256']
 protocol={'schema':'real_polygon_reference_query_development.v1','scene':a.scene,'field_sha256':records,'geometry_sha256':g['geometry_sha256'],'labels':[{'path':str(x),'sha256':sha256_file(x)} for x in sorted((a.label_root/a.scene).glob('frame_*.json'))],
   'reference_selection':'first annotated frame per category; remaining annotated frames are targets','query_contract':'text verbatim; image masked crop with3px padding and gray127 outside; prompt one positive and one negative point with registered frozen geometry','query_mask_or_points_supplied_from_reference_gt':True,'target_masks_used_for_query_encoding':False,'annotations_define_evaluation_cohort':True,'target_rgb_encoded_for_its_own_query':False,'source_training_assets_forbidden':True,'source_semantic_train_frames':info['train_frames'],'target_width':480,'threshold':.5,'refine_prompts':a.refine_prompts,'condition_images':a.condition_images,'image_conditioning_input':'same authorized reference positive/negative pixels as prompt, in addition to reference image crop','seed_support_policy':'full alpha > 1e-8 and modality observed/full contribution > .5','conditioning_implementation_sha256':sha256_file(condition_native_image_query.__code__.co_filename),'prompt_refinement_implementation_sha256':sha256_file(condition_signed_query.__code__.co_filename),'minimal_prompt_refinement':a.minimal_prompt_refinement,'official_benchmark_protocol':False,'ground_truth_domain':'2D category polygon unions; no 3D ground truth','prompt_input_count':{'positive':1,'negative':1},'prompt_registration_required':True,'target_category_absence_evaluated':False,'instance_identity_verified':False,'metric_contract_names':{'text':'verbatim_text_category_union','image':'GT_masked_reference_image_category_union','prompt':'GT_two_point_registered_category_union','prompt_refined':'GT_two_point_registered_category_union_with_transient_query_optimization'},'instance_identity_assumption':'same category across frames, union of category polygons; not independently verified physical instance tracks','implementation_sha256':sha256_file(__file__),'modality_support_implementation_sha256':sha256_file('radio_gs/rendering/modality_support.py')}
 a.output.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 annotations,categories,sh,sw=load_labels(a.label_root,a.scene)
 if set(annotations)&set(info['train_frames']):raise ValueError('reference/evaluation images overlap semantic training')
 qh,qw=auth['metadata']['feature_height'],auth['metadata']['feature_width'];height=round(sh*480/sw)
 references=[];target_tasks=[];hits_cache={}
 for category in categories:
  frames=[f for f,o in sorted(annotations.items()) if any(x['category']==category for x in o)]
  if len(frames)<2:continue
  frame=frames[0];index=len(references);image_path=Path(source['image_dir'])/f'frame_{frame:05d}.jpg';im=Image.open(image_path).convert('RGB')
  native=ground_truth_masks(annotations[frame],[category],im.height,im.width,sh,sw)[category].numpy();ys,xs=np.where(native)
  rgb=np.asarray(im).copy();rgb[~native]=127;box=(max(0,int(xs.min())-3),max(0,int(ys.min())-3),min(im.width,int(xs.max())+4),min(im.height,int(ys.max())+4));crop=Image.fromarray(rgb).crop(box)
  if frame not in hits_cache:
   camera=prior.cameras[frame];k=camera.intrinsic.to(a.device).clone();k[0]*=qw/camera.width;k[1]*=qh/camera.height
   renderer=FeatureFieldRenderer(qh,qw,float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2]),near_plane=.001*prior.scale,far_plane=100*prior.scale).to(a.device)
   hits_cache[frame]=rasterize_single_view_contributions(geometry,renderer,camera.world_to_camera.to(a.device),height=qh,width=qw)
  hit=hits_cache[frame];mass=torch.zeros(qh*qw,device=a.device).scatter_add_(0,hit['pixel_ids'].long(),hit['weights'].float()).reshape(qh,qw)
  mask=ground_truth_masks(annotations[frame],[category],qh,qw,sh,sw)[category].to(a.device);positions=[];seedrows=[]
  for signed in (mask,~mask):
   legal=signed&(mass>.1)
   if not legal.any():break
   pixel=int(distance_transform_edt(legal.cpu().numpy()).argmax());positions.append(pixel)
   selected=torch.where(hit['pixel_ids']==pixel)[0];seedrows.append(int(hit['gaussian_ids'][selected[hit['weights'][selected].argmax()]]))
  references.append({'category':category,'frame':frame,'image_path':image_path,'image_sha256':sha256_file(image_path),'crop':crop,'pixels':positions,'seedrows':seedrows})
  target_tasks.extend({'reference':index,'target_frame':f,'category':category} for f in frames[1:])
 if not references:raise ValueError('no cross-view reference tasks')
 encoder=NativeThreeQueryEncoder(identity,a.device);encoded={'text':encoder.encode_text([x['category'] for x in references]).half().float()}
 encoded['image']=torch.cat([encoder.encode_image(x['crop']) for x in references]).half().float()
 encoder.text_model=None;encoder.vision=None;gc.collect();torch.cuda.empty_cache()
 lo=geometry.xyz.amin(0);extent=(geometry.xyz.amax(0)-lo).clamp_min(1e-6);prompt=[]
 for r in references:
  if len(r['pixels'])!=2:prompt.append(None);continue
  prompt.append(encoder.encode_prompt(r['image_path'],r['pixels'],geometry.xyz[r['seedrows']],lo,extent,qh,qw).half().float())
 encoder.dino=None;gc.collect();torch.cuda.empty_cache()
 modalities=['text','image','prompt']+(['prompt_refined'] if a.refine_prompts else [])+(['image_conditioned'] if a.condition_images else [])
 results={}
 with torch.no_grad():
  for run,digest in records.items():
   field,payload=load_element_retrieval_field(a.root/run/'field.pt',expected_sha256=digest,geometry=geometry,expected_query_encoder=identity,device=a.device)
   if field.image_observed is not None and a.condition_images:raise ValueError('legacy native-query conditioning has no extended-support contract')
   if payload['query_context']['prompt_raster_size']!=[qh,qw]:raise ValueError('prompt raster differs from field')
   normal=payload['query_context']['prompt_world_normalization']
   if not torch.allclose(normal['minimum'].to(lo),lo) or not torch.allclose(normal['extent'].to(extent),extent):raise ValueError('prompt normalization differs from field')
   scores={m:field.retrieve(encoded[m],m)['scores'] for m in ('text','image')}
   scores['prompt']=torch.cat([field.retrieve(q,'prompt')['scores'] if q is not None else torch.zeros(len(geometry.xyz),1,device=a.device) for q in prompt],dim=1)
   refinements=[]
   if a.refine_prompts:
    refined=[]
    for qi,(reference,q) in enumerate(zip(references,prompt)):
     if q is None:
      refined.append(scores['prompt'][:,qi:qi+1]);refinements.append({'available':False,'reason':'no registered source geometry'});continue
     hit=hits_cache[reference['frame']];seed_rows=[];seed_weights=[]
     for pixel in reference['pixels']:
      select=hit['pixel_ids']==pixel;seed_rows.append(hit['gaussian_ids'][select].long());seed_weights.append(hit['weights'][select].float())
     conditioned=condition_signed_query(field,q,seed_rows,seed_weights,[1.,0.],stop_when_satisfied=a.minimal_prompt_refinement)
     refined.append(retrieve_latent_query(field,conditioned['latent_query'])['scores'])
     refinements.append({k:v.tolist() if torch.is_tensor(v) else v for k,v in conditioned.items() if k!='latent_query'})
    scores['prompt_refined']=torch.cat(refined,dim=1)
   image_conditioning=[]
   if a.condition_images:
    conditioned_images=[]
    for qi,reference in enumerate(references):
     if len(reference['pixels'])!=2:
      conditioned_images.append(scores['image'][:,qi:qi+1]);image_conditioning.append({'available':False,'reason':'no registered source geometry'});continue
     hit=hits_cache[reference['frame']];seed_rows=[];seed_weights=[]
     for pixel in reference['pixels']:
      select=hit['pixel_ids']==pixel;seed_rows.append(hit['gaussian_ids'][select].long());seed_weights.append(hit['weights'][select].float())
     try:
      conditioned=condition_native_image_query(field,encoded['image'][qi:qi+1],seed_rows,seed_weights,[1.,0.])
      conditioned_images.append(field.retrieve(conditioned['native_query'],'image')['scores'])
      image_conditioning.append({'available':True,**{k:v.tolist() if torch.is_tensor(v) else v for k,v in conditioned.items() if k!='native_query'}})
     except SignedEvidenceUnavailable as error:
      conditioned_images.append(scores['image'][:,qi:qi+1]);image_conditioning.append({'available':False,'reason':str(error),'missing_seed_index':error.seed_index})
    scores['image_conditioned']=torch.cat(conditioned_images,dim=1)
   supports=[field.observed_for('image' if m=='image_conditioned' else 'prompt' if m=='prompt_refined' else m) for m in modalities]
   shared_support=all(torch.equal(supports[0],x) for x in supports[1:])
   colors=torch.cat([scores[m] for m in modalities]+[x.float()[:,None] for x in (supports[:1] if shared_support else supports)],dim=1)
   rows=[]
   for frame in sorted({e['target_frame'] for e in target_tasks}):
    camera=prior.cameras[frame];k=camera.intrinsic.to(a.device).clone();k[0]*=480/camera.width;k[1]*=height/camera.height
    image,alpha,_=rasterization(means=geometry.xyz,quats=geometry.quaternions,scales=geometry.scales,opacities=geometry.opacities[:,0],colors=colors,viewmats=camera.world_to_camera.to(a.device)[None],Ks=k[None],width=480,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
    from radio_gs.rendering.modality_support import supported_modality_response
    targets=ground_truth_masks(annotations[frame],categories,height,480,sh,sw)
    for e in [x for x in target_tasks if x['target_frame']==frame]:
     truth=targets[e['category']].to(a.device);r=e['reference'];row={**e,'reference_frame':references[r]['frame'],'prompt_available':len(references[r]['pixels'])==2}
     for mi,m in enumerate(modalities):
      p=supported_modality_response(image[0],alpha[0,...,0],query_index=r,modality_index=mi,num_queries=len(references),num_modalities=len(modalities),shared_support=shared_support);iou,_,_=binary_iou(p>.5,truth);row[m]={'iou':iou,**binary_ranking_metrics(p,truth)}
     rows.append(row)
   metrics={m:{k:sum(r[m][k] for r in rows)/len(rows) for k in ['iou','average_precision','auroc']} for m in modalities}
   if any(p.grad is not None for p in field.parameters()):raise ValueError('query execution wrote field gradients')
   results[run]={'field_sha256':digest,'tasks_per_modality':len(rows),'unavailable_prompt_tasks':sum(not row['prompt_available'] for row in rows),'macro':metrics,'observations':rows,'prompt_refinement':refinements,'image_conditioning':image_conditioning,'support_elements':{m:int(v.sum()) for m,v in zip(modalities,supports)}};print(run,metrics,flush=True)
 geometry.assert_binding(auth['metadata']['frozen_geometry_binding'])
 a.output.write_text(json.dumps({'protocol_sha256':sha256_file(a.output.with_suffix('.protocol.json')),'protocol':protocol,'references':[{k:str(v) if isinstance(v,Path) else v for k,v in r.items() if k!='crop'} for r in references],'runs':results,'geometry_unchanged':True},indent=2)+'\n')

if __name__=='__main__':main()
