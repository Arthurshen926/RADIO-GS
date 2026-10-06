"""Public NVOS positive-only 2D scribble -> reference-only SAM masks and image-query crops.

This is query-input construction, not NVOS target segmentation evaluation.
"""
import argparse,json,sys,os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from radio_gs.querying.interactive_2d import validate_2d_prompt,scribble_points,select_prompt_mask,masked_query_crop
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model,set_requested_cuda_device,sam3_autocast_context,sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json'));p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/sam3_modelscope/sam3.pt'));p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError(a.output)
 d=json.loads(a.manifest.read_text());forbidden=set()
 for s in d['scenes']:
  forbidden.add(str(Path(s['prompt']['negative_path']).resolve()))
  for r in s['frames']:
   if r['frame_id'] in s['evaluation_frame_ids']:
    for k in ['rgb_path','ground_truth','gt_mask_path']:
     if r.get(k):forbidden.add(str(Path(r[k]).resolve()))
 def forbid_target(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and str(Path(os.fsdecode(args[0])).resolve()) in forbidden:raise RuntimeError('NVOS target RGB/GT forbidden while constructing reference query')
 sys.addaudithook(forbid_target);a.output.mkdir(parents=True);set_requested_cuda_device(a.device)
 model=_load_sam3_model(checkpoint_path=str(a.checkpoint),device=a.device,confidence_threshold=0,dtype='float32',resolution=1008,point_only=True,build_on_cpu=True)
 rows=[]
 for s in d['scenes']:
  prompt=s['prompt'];reference=next(r for r in s['frames'] if r['frame_id']==prompt['frame_id']);image_path=Path(reference['rgb_path']);image=Image.open(image_path).convert('RGB');evidence={};inputs={}
  for name in ['positive']:
   path=Path(prompt[name+'_path']);digest=sha256_file(path);assert digest==d['protocol']['prompt_asset_sha256'][s['scene_id']][name];values=np.asarray(Image.open(path));evidence[name]=np.any(values>127,axis=2) if values.ndim==3 else values>127;inputs[name]={'path':str(path),'sha256':digest}
  evidence['negative']=np.zeros_like(evidence['positive']);height,width=evidence['positive'].shape;points,labels=scribble_points(evidence['positive'],evidence['negative']);kwargs=validate_2d_prompt(width,height,points=points,labels=labels)
  with torch.inference_mode(),sam3_autocast_context(a.device,torch.bfloat16):
   state=model.set_image(image.resize((width,height),Image.Resampling.LANCZOS));masks,quality,_=model.model.predict_inst(state,**kwargs,multimask_output=True)
  mask,selection=select_prompt_mask(np.asarray(masks).astype(bool),np.asarray(quality),**evidence)
  folder=a.output/s['scene_id'];folder.mkdir();mask_path=folder/'reference_sam_mask.png';Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
  result={'scene':s['scene_id'],'reference_frame':prompt['frame_id'],'reference_rgb':{'path':str(image_path),'sha256':sha256_file(image_path)},'official_scribbles':inputs,'negative_scribble_opened':False,'input_contract':'nvos-ludvig-online-all-view-eight-v1','prompt_raster_wh':[width,height],'original_rgb_wh':list(image.size),'points_xy':points.tolist(),'labels':labels.tolist(),'selection':selection,'mask_sha256':sha256_file(mask_path),'mask':str(mask_path),'complex_rope_buffers_preserved':True,'model_cast_implementation_sha256':sha256_file(Path('radio_gs/scripts/build_sam3_foundation_cache.py')),'reference_mask_is_gt':False,'target_reads_forbidden':True}
  if mask.any():
   native_mask=np.asarray(Image.fromarray(mask).resize(image.size,Image.Resampling.NEAREST));crop,box=masked_query_crop(image,native_mask);crop_path=folder/'image_query.png';crop.save(crop_path);result.update(image_query_crop={'path':str(crop_path),'sha256':sha256_file(crop_path),'box_xyxy':list(box)},query_available=True)
  else:result.update(query_available=False,reason='SAM returned empty reference mask; do not fabricate a crop')
  rows.append(result);(folder/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(s['scene_id'],selection['selected'],flush=True)
 result={'schema':'nvos_positive_reference_sam_image_query.v1','manifest_sha256':sha256_file(a.manifest),'sam_checkpoint_sha256':sha256_file(a.checkpoint),'implementation_sha256':sha256_file(Path(__file__)),'prompt_implementation_sha256':sha256_file(Path('radio_gs/querying/interactive_2d.py')),'query_route':'official positive-only 2D scribble -> reference SAM mask -> native image encoder -> scene field image readout','not_a_target_prediction':True,'not_a_benchmark_score':True,'target_rgb_or_gt_opened':False,'scenes':rows};(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
