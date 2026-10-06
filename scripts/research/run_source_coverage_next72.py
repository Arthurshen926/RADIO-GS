"""Prepare complete-source teachers; coverage64 is an explicit intermediate cohort."""
import json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from radio_gs.utils.immutable_artifacts import sha256_file
PY='/root/miniconda3/envs/cybersim_agent/bin/python';SAM='/root/miniconda3/envs/sam3/bin/python'
OUT=Path('output/source_coverage_next72')
def run(item):
 gpu,scene=item;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='.',OMP_NUM_THREADS='4');out=OUT/scene;out.mkdir(parents=True,exist_ok=True)
 with (out/'execution.log').open('w') as log:
  cmd=[PY,'scripts/research/prepare_source_coverage_next72.py','--scene',scene];subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  cohorts=json.loads((out/'cohorts.json').read_text());authority=out/'all.json';commands=[]
  for cohort in ['coverage64','all']:
   cmd=[SAM,'scripts/audits/run_lerf_source_stage.py','--module','radio_gs.scripts.build_sam3_multiscale_hierarchy_cache','--source-authority',str(authority),'--source-authority-sha256',sha256_file(authority),'--image-ids',','.join(f'frame_{f:05d}' for f in cohorts['frames'][cohort]),'--output-root',str(out/'source_sam'),'--manifest-name',f'manifest_{cohort}.json','--checkpoint-path','checkpoints/sam3_modelscope/sam3.pt','--checkpoint-sha256','9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e','--device','cuda:0','--skip-existing']
   commands.append(dict(cohort=cohort,command=cmd,status='running'));(out/'execution.json').write_text(json.dumps(commands,indent=2));rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;commands[-1].update(status='complete' if rc==0 else 'failed',exit_code=rc);(out/'execution.json').write_text(json.dumps(commands,indent=2))
   if rc:raise RuntimeError((scene,cohort,rc))
 return scene
if __name__=='__main__':
 with ThreadPoolExecutor(4) as pool:
  list(pool.map(run,[(0,'figurines'),(1,'ramen'),(2,'teatime'),(3,'waldo_kitchen')]))
