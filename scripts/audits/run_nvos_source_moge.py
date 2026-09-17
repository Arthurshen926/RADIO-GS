"""Run exact MoGe on one sealed NVOS source cohort with target-file denial."""
import argparse,json,os,sys
from pathlib import Path
from radio_gs.v4.geometry.batch_moge3 import run


def main():
 p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--scene',required=True);p.add_argument('--device',default='cuda:2');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 records=json.loads(a.sources.read_text())['scenes'];r=next(x for x in records if x['scene']==a.scene)
 authority=json.loads(Path(r['source_authority']).read_text());manifest=json.loads(Path(authority['manifest']).read_text());forbidden=set()
 for s in manifest['scenes']:
  for f in s['frames']:
   if f['frame_id'] in s['evaluation_frame_ids']:
    for k in ('rgb_path','annotation_rgb_path','ground_truth','gt_mask_path'):
     if f.get(k):forbidden.add(str(Path(f[k]).resolve()))
  for k in ('positive_path','negative_path','visualization_path'):
   if s['prompt'].get(k):forbidden.add(str(Path(s['prompt'][k]).resolve()))
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and str(Path(os.fsdecode(args[0])).resolve()) in forbidden:raise RuntimeError('target/semantic prompt input forbidden during geometry construction')
 sys.addaudithook(guard)
 result=run(argparse.Namespace(scene_root=r['scene_root'],scene_label=a.scene,source_authority=r['source_authority'],output_dir=str(a.output),fov_x_degrees=r['fov_x_degrees'],model_id='Ruicheng/moge-3-vitl',device=a.device,resolution_level=9,refine_steps=3,disable_fp16=False,geometry_dtype='float32',shard_id=0,num_shards=1))
 print(json.dumps({'records':len(result['records']),'target_or_prompt_inputs_used':False}))
if __name__=='__main__':main()
