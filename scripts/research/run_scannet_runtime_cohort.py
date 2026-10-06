"""Complete every declared category scene, then report both label domains."""
import argparse,json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpus',default='2,3');p.add_argument('--output',default='output/scannet_runtime_integration');p.add_argument('--skip-first-inference',action='store_true');a=p.parse_args();cohort=json.loads(Path('configs/consolidated/scannet/cohort.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True);gpus=a.gpus.split(',')
 def lane(gpu,items):
  for item in items:
   root=out/item['scene'];root.mkdir(parents=True,exist_ok=True);commands=[]
   if not (a.skip_first_inference and item['scene']=='scene0000_00'):commands.append([PY,'scripts/research/run_scannet_query.py','--scene',item['scene_config'],'--query',item['query_config'],'--output',str(root),'--gpu',gpu])
   commands.append([PY,'scripts/research/evaluate_scannet_runtime.py','--readout',str(root/'typed_readout.pt'),'--evaluation',item['evaluation_config'],'--output',str(root/'evaluation')])
   (root/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
   with (root/'run.log').open('w') as log:
    for command in commands:subprocess.run(command,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,check=True)
   print(item['scene'],'complete',flush=True)
 with ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(lambda x:lane(*x),[(g,cohort['scenes'][i::len(gpus)]) for i,g in enumerate(gpus)]))
 reports={r['scene']:json.loads((out/r['scene']/'evaluation/report.json').read_text()) for r in cohort['scenes']}
 summary=dict(scenes=reports,scene_macro={domain:{arm:{s:float(np.mean([v[domain][arm][s]['miou'] for v in reports.values()])) for s in ['19','15','10']} for arm in ['map','source']} for domain in ['point_metrics','gaussian_volume_pseudolabel_metrics']},fresh_predictions=True,map_rebuilt=False,historical_predictions_read=False,official_protocol_equivalence_verified=False)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary['scene_macro'],flush=True)
if __name__=='__main__':main()
