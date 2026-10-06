"""Consume completed source cohorts; run matched-old and expanded-native pilots."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
def run(item):
 gpu,scene=item;out=Path('output/source_coverage_next72')/scene;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.');key='lerf_'+scene
 while not (out/'execution.json').exists():time.sleep(10)
 for cohort in ['old','coverage64','all']:
  if (Path('output/expanded_semantic_next72')/cohort/'old_extent'/key/'anchored/pipeline_calibrated/evaluation.json').exists():continue
  frames=json.loads((out/'cohorts.json').read_text())['frames'][cohort]
  while not all((out/'source_sam'/f'frame_{f:05d}.pt').exists() for f in frames):
   status=json.loads((out/'execution.json').read_text())
   if any(r['status']=='failed' for r in status):raise RuntimeError((scene,'source producer failed'))
   time.sleep(10)
  commands=[[PY,'scripts/research/encode_source_regions_next72.py','--scene',scene,'--cohort',cohort],[PY,'scripts/research/rebuild_expanded_semantic_next72.py','--key',key,'--cohort',cohort]]
  for name in ['refine','align']:commands.append([PY,f'scripts/research/{name}_joint_semantic_next66.py','--key',key,'--arm',cohort,'--recipe-root','output/expanded_semantic_next72'])
  selected=f'output/expanded_semantic_next72/{cohort}/old_extent'
  commands.append([PY,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--mode','seeded','--cohort','old','--evidence-base','output/typed_evidence_next44','--semantic-base',f'output/expanded_semantic_next72/{cohort}/calibrated','--output-base',selected])
  commands.append([PY,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',selected,'--readout','anchored','--pipeline','--calibrated','--element-results',f'{selected}/{key}/typed_readout.pt'])
  rows=[]
  with open(f'/tmp/radio-gs-next72-gpu{gpu}.lock','w') as lease,(out/(cohort+'_semantic.log')).open('w') as log:
   fcntl.flock(lease,fcntl.LOCK_EX)
   for cmd in commands:
    rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(command=cmd,exit_code=rc));(out/(cohort+'_semantic_execution.json')).write_text(json.dumps(rows,indent=2))
    if rc:raise RuntimeError((scene,cohort,cmd[1],rc))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scenes',default='figurines,ramen,teatime,waldo_kitchen');a=p.parse_args()
 with ThreadPoolExecutor(4) as pool:list(pool.map(run,[(4+i%2,s) for i,s in enumerate(a.scenes.split(','))]))
