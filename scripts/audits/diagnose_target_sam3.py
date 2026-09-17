"""Privileged target-RGB SAM3 localization diagnostic; never source supervision."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model,set_requested_cuda_device,sam3_autocast_context
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks,binary_iou
# SAM3 runtime intentionally has no Gaussian renderer; load the pure metric module
# without importing training.__init__ and its unrelated gsplat dependency.
import importlib.util
_metric_path=Path(__file__).resolve().parents[2]/'radio_gs/training/retrieval_metrics.py'
_spec=importlib.util.spec_from_file_location('diagnostic_retrieval_metrics',_metric_path)
_metrics=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_metrics)
binary_ranking_metrics=_metrics.binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists() or a.output.with_suffix('.protocol.json').exists():raise FileExistsError(a.output)
 source=json.loads((a.root/'source_frames.json').read_text());labels=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label');annotations,categories,sh,sw=load_labels(labels,a.scene);height=round(sh*480/sw);checkpoint=Path('checkpoints/sam3_modelscope/sam3.pt')
 protocol={'schema':'target_sam3_diagnostic.v1','target_rgb_used':True,'target_gt_used_for_model_input':False,'evaluation_categories_used_as_queries':True,'training_or_deployment_result':False,'confidence_threshold':.7,'mask_threshold':.5,'width':480,'checkpoint_sha256':sha256_file(checkpoint),'implementation_sha256':sha256_file(__file__),'labels':{str(x):sha256_file(x) for x in sorted((labels/a.scene).glob('*.json'))}}
 a.output.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');set_requested_cuda_device(a.device);model=_load_sam3_model(checkpoint_path=str(checkpoint),device=a.device,confidence_threshold=.7,dtype='float32',resolution=1008,build_on_cpu=True);rows=[];images={}
 with torch.inference_mode():
  for frame,objects in sorted(annotations.items()):
   path=Path(source['image_dir'])/f'frame_{frame:05d}.jpg';images[str(path)]=sha256_file(path)
   with sam3_autocast_context(a.device,torch.bfloat16):state=model.set_image(Image.open(path).convert('RGB'))
   predictions={}
   # Run the same full category vocabulary on every image, regardless of GT presence.
   for category in categories:
    model.reset_all_prompts(state)
    with sam3_autocast_context(a.device,torch.bfloat16):out=model.set_text_prompt(category,state)
    prob=out['masks_logits'];score=F.interpolate(prob.float().amax(0,keepdim=True),size=(height,480),mode='bilinear',align_corners=False)[0,0] if len(prob) else torch.zeros(height,480,device=a.device);predictions[category]=score
   targets=ground_truth_masks(objects,categories,height,480,sh,sw)
   for category,score in predictions.items():
    target=targets[category].to(a.device)
    if not target.any():continue
    iou,_,_=binary_iou(score>.5,target);rows.append({'frame':frame,'category':category,'iou':iou,**binary_ranking_metrics(score,target)})
   print('target',frame,flush=True)
 macro={k:sum(x[k] for x in rows)/len(rows) for k in ('iou','average_precision','auroc')};a.output.write_text(json.dumps({'protocol':protocol,'images':images,'macro':macro,'observations':rows},indent=2)+'\n');print(macro,flush=True)
if __name__=='__main__':main()
