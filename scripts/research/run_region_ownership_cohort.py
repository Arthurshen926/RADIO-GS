"""Full matched ScanNet8 and LERF4 ownership reconstruction and frozen evaluation."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['scannet','lerf'],required=True);p.add_argument('--gpus',default='0,1,2,3,4,5');a=p.parse_args();gpus=a.gpus.split(',')
 rows=json.loads(Path('configs/consolidated/scannet/cohort.json').read_text())['scenes'] if a.dataset=='scannet' else [dict(scene=s) for s in ['ramen','figurines','teatime','waldo_kitchen']]
 def worker(index):
  gpu=gpus[index]
  for row in rows[index::len(gpus)]:
   name=row['scene'];out=Path('output/native_region_ownership')/name;out.mkdir(parents=True,exist_ok=True);env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpu)
   def run(cmd,log):
    with (out/log).open('w') as f:subprocess.run([PY,*map(str,cmd)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   if a.dataset=='scannet':
    run(['scripts/research/rebuild_native_region_ownership.py','--scene',name,'--gpu',gpu,'--output',out/'map'],'build.log')
    run(['-m','radio_gs.pipeline','--scene',out/'map/scene.json','--queries',row['query_config'],'--access','source-assisted','--output',out/'query','--gpu',gpu],'query.log')
    run(['scripts/research/evaluate_scannet_runtime.py','--readout',out/'query/category/typed_readout.pt','--evaluation',row['evaluation_config'],'--output',out/'evaluation','--point-only'],'evaluation.log')
   else:
    run(['-m','radio_gs.pipeline.build_map','--scene',f'configs/pipeline/reconstruction/{name}.json','--regions',f'output/system_map_rebuild/{name}/regions','--output',out/'map','--region-policy','finest'],'build.log')
    queries=f'configs/consolidated/{name}.integration-queries.json' if name=='ramen' else f'configs/consolidated/{name}.queries.json'
    run(['-m','radio_gs.pipeline','--scene',out/'map/scene.json','--queries',queries,'--access','map-only','--output',out/'query','--gpu',gpu],'query.log')
    run(['scripts/research/evaluate_consolidated_query.py','--scene',out/'map/scene.json','--readout',out/'query/native/response.pt','--direct-map','--output',out/'evaluation'],'evaluation.log')
   print(name,'complete',flush=True)
 with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(worker,range(len(gpus))))
if __name__=='__main__':main()
