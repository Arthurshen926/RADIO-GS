"""Execute one source-only build stage with explicit NVOS target/prompt denial."""
import argparse, json, os, runpy, sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--module',action='store_true');p.add_argument('entry');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
m=json.loads(Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json').read_text())
s=next(x for x in m['scenes'] if x['scene_id']==a.scene);forbidden=set()
for f in s['frames']:
 if f['frame_id'] in s['evaluation_frame_ids']:
  for k in ('rgb_path','annotation_rgb_path','gt_mask_path','ground_truth'):
   if f.get(k):forbidden.add(str(Path(f[k]).resolve()))
for k in ('positive_path','negative_path','visualization_path'):
 if s['prompt'].get(k):forbidden.add(str(Path(s['prompt'][k]).resolve()))
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)) and str(Path(os.fsdecode(args[0])).resolve()) in forbidden:
  raise RuntimeError('source stage attempted to open NVOS target or prompt asset')
sys.addaudithook(guard);sys.argv=[a.entry,*a.arguments]
if a.module:runpy.run_module(a.entry,run_name='__main__')
else:runpy.run_path(a.entry,run_name='__main__')
