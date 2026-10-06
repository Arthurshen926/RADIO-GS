"""Equal-evidence early/late fusion, complete scene evaluation without tuning."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['scannet','lerf'],required=True);p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--scenes',nargs='+');a=p.parse_args();gpus=a.gpus.split(',')
 rows=json.loads(Path('configs/consolidated/scannet/cohort.json').read_text())['scenes'] if a.dataset=='scannet' else [dict(scene=s) for s in ['ramen','figurines','teatime','waldo_kitchen']]
 if a.scenes:rows=[r for r in rows if r['scene'] in a.scenes]
 def worker(i):
  for row in rows[i::len(gpus)]:
   name=row['scene'];base=f'configs/pipeline/scannet_native_calibrated/{name}.scene.json' if a.dataset=='scannet' else f'output/system_map_rebuild/{name}/complete/scene.json'
   for mode in ['early_mean','late_logmeanexp']:
    out=Path('output/native_scope_pair')/name/mode;out.mkdir(parents=True,exist_ok=True);scene=json.loads(Path(base).read_text());scene.update(native_scope_field=f'output/native_region_ownership/{name}/map/native_field.pt',scope_readout=mode)
    (out/'scene.json').write_text(json.dumps(scene,indent=2)+'\n');env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[i])
    def run(cmd,log):
     with (out/log).open('w') as f:subprocess.run([PY,*map(str,cmd)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    queries=row['query_config'] if a.dataset=='scannet' else (f'configs/consolidated/{name}.integration-queries.json' if name=='ramen' else f'configs/consolidated/{name}.queries.json')
    run(['-m','radio_gs.pipeline','--scene',out/'scene.json','--queries',queries,'--access','source-assisted' if a.dataset=='scannet' else 'map-only','--output',out/'query','--gpu',gpus[i]],'query.log')
    if a.dataset=='scannet':run(['scripts/research/evaluate_scannet_runtime.py','--readout',out/'query/category/typed_readout.pt','--evaluation',row['evaluation_config'],'--output',out/'evaluation','--point-only'],'evaluation.log')
    else:run(['scripts/research/evaluate_consolidated_query.py','--scene',out/'scene.json','--readout',out/'query/native/response.pt','--direct-map','--output',out/'evaluation'],'evaluation.log')
    print(name,mode,'complete',flush=True)
 with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(worker,range(len(gpus))))
if __name__=='__main__':main()
