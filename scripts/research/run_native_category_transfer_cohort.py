"""Complete matched-eight reconstruction/readout comparisons or public replay."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['heads','scope','joint','public'],required=True);p.add_argument('--gpus',default='0,1,2,3,4,5');a=p.parse_args()
 root=Path('output/native_category_transfer');root.mkdir(exist_ok=True)
 cohort='configs/pipeline/scannet_native_calibrated/cohort.json' if a.mode=='public' else 'configs/consolidated/scannet/cohort.json'
 rows=json.loads(Path(cohort).read_text())['scenes'];gpus=a.gpus.split(',');py='/root/miniconda3/envs/cybersim_agent/bin/python'
 env=dict(os.environ,PYTHONPATH=str(Path.cwd()))
 def worker(index):
  for row in rows[index::len(gpus)]:
   scene=row['scene'];out=root/scene;out.mkdir(exist_ok=True)
   with (out/(a.mode+'.log')).open('w') as log:
    if a.mode=='heads':
     cmd=[py,'scripts/research/run_native_category_transfer.py','--scene',scene,'--gpu',gpus[index],'--output',str(out)];arms=['transferred','source_refit']
    elif a.mode=='joint':
     cmd=[py,'scripts/research/run_native_categorical_evidence.py','--scene',scene,'--gpu',gpus[index]];arms=['joint_category']
    elif a.mode=='scope':
     cmd=[py,'scripts/research/run_native_scope_transfer.py','--scene',scene,'--gpu',gpus[index]];arms=['scope_transfer']
    else:
     cmd=[py,'-m','radio_gs.pipeline','--scene',row['scene_config'],'--queries',row['query_config'],'--access','source-assisted','--gpu',gpus[index],'--output',str(out/'public')];arms=['public/category']
    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    for arm in arms:
     subprocess.run([py,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/arm/'typed_readout.pt'),'--evaluation',row['evaluation_config'],'--output',str(out/arm/'evaluation'),'--point-only'],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
   print(scene,a.mode,'complete',flush=True)
 with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(worker,range(len(gpus))))
if __name__=='__main__':main()
