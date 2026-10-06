"""Frozen admitted source evidence: isolate the effect of region ownership."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',default=['ramen','figurines','teatime','waldo_kitchen']);p.add_argument('--gpus',default='0,1,2,3');a=p.parse_args();gpus=a.gpus.split(',')
 def worker(i):
  for name in a.scenes[i::len(gpus)]:
   out=Path('output/native_region_ownership')/name;queries=f'configs/consolidated/{name}.integration-queries.json' if name=='ramen' else f'configs/consolidated/{name}.queries.json'
   evidence=json.loads(Path(f'output/system_map_rebuild/{name}/source_query/text/identity_evidence/inputs.json').read_text())['detections']
   env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[i])
   commands=[(['-m','radio_gs.pipeline','--scene',out/'map/scene.json','--queries',queries,'--access','source-assisted','--native-response',out/'query/native/response.pt','--source-evidence',evidence,'--provider-admitted','--output',out/'source_query','--gpu',gpus[i]],'source.log'),(['scripts/research/evaluate_consolidated_query.py','--scene',out/'map/scene.json','--readout',out/'source_query/typed_readout.pt','--output',out/'source_evaluation'],'source_evaluation.log')]
   for cmd,log in commands:
    with (out/log).open('w') as f:subprocess.run([PY,*map(str,cmd)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   print(name,'source complete',flush=True)
 with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(worker,range(len(gpus))))
if __name__=='__main__':main()
