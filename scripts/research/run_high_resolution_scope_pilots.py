import os,json,subprocess,shutil,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
python='/root/miniconda3/envs/cybersim_agent/bin/python'
def run(pair):
 gpu,scene=pair;key='lerf_'+scene;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
 base=Path('output/high_resolution_scopes_next38');out=base/key;out.mkdir(parents=True,exist_ok=True)
 with (out/'execution.log').open('w') as log:
  command=[python,'scripts/research/fit_rendered_scope_memberships.py','--key',key,'--source','output/scope_complete_next36','--output',str(base),'--evidence-root','output/high_resolution_scope_evidence_next38/'+key]
  rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  if rc:return dict(key=key,exit_code=rc)
  d=torch.load(out/'field.pt',weights_only=False);d.update(scope_native_dictionary=True,native_modalities=['text'],scope_modalities=['text','image'],scope_readout='composition',composition_fixed_cores=True,scope_after_prompt=False);torch.save(d,out/'field.pt');shutil.copyfile(Path('output/scope_query_modes_next34')/key/'calibration.pt',out/'calibration.pt')
  commands=[[python,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(base),'--readout','anchored','--pipeline','--calibrated'],[python,'scripts/research/evaluate_scope_reference_images.py','--key',key,'--base',str(base)],[python,'scripts/research/measure_scope_hypothesis_retrieval.py','--scene',scene,'--base',str(base)]]
  for command in commands:
   rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
   if rc:break
 return dict(key=key,exit_code=rc)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scenes',default='figurines,ramen');p.add_argument('--gpus',default='4,5');a=p.parse_args()
 pairs=list(zip(map(int,a.gpus.split(',')),a.scenes.split(',')))
 with ThreadPoolExecutor(len(pairs)) as pool:results=list(pool.map(run,pairs))
 receipt=Path('output/high_resolution_scopes_next38/execution.json')
 previous=json.loads(receipt.read_text()) if receipt.exists() else []
 updated={r['key']:r for r in previous+results};receipt.write_text(json.dumps(list(updated.values()),indent=2)+'\n');print(results)
