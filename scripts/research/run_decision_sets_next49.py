import os,json,subprocess,argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
parser=argparse.ArgumentParser();parser.add_argument('--arm',default='baseline');parser.add_argument('--gpus',default='0,1,2,3');args=parser.parse_args();arm=args.arm;gpus=args.gpus.split(',')
def run(job):
    gpu,scene=job;key='lerf_'+scene;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
    root=Path('output/decision_sets_next49');root.mkdir(parents=True,exist_ok=True)
    commands=[[PYTHON,'scripts/research/decode_decision_sets_next49.py','--key',key,'--arm',arm],[PYTHON,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root/arm),'--readout','anchored','--pipeline','--calibrated','--element-results',str(root/arm/key/'elements.pt')]];rows=[]
    with (root/(scene+'_'+arm+'.log')).open('w') as log:
        for c in commands:
            rc=subprocess.run(c,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(scene=scene,command=c,exit_code=rc));print(scene,c[1],rc,flush=True)
            if rc:break
    return rows
rows=[];scenes=['figurines','ramen','teatime','waldo_kitchen']
for start in range(0,len(scenes),len(gpus)):
    with ThreadPoolExecutor(len(gpus)) as pool:rows+=sum(pool.map(run,[(gpus[i],s) for i,s in enumerate(scenes[start:start+len(gpus)])]),[])
(Path('output/decision_sets_next49')/('execution_'+arm+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
