import json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PY='/root/miniconda3/envs/cybersim_agent/bin/python'

def run(job):
    gpu,key,arm=job;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.');out=Path('output/subset_readout_next45');commands=[[PY,'scripts/research/fit_subset_readout_next45.py','--key',key,'--arm',arm],[PY,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out/arm),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/arm/key/'elements.pt')]];rows=[]
    with (out/(key+'_'+arm+'.log')).open('w') as f:
        for c in commands:
            rc=subprocess.run(c,env=env,stdout=f,stderr=subprocess.STDOUT).returncode;rows.append(dict(key=key,arm=arm,command=c,exit_code=rc));print(key,arm,c[1],rc,flush=True)
            if rc:break
    return rows
if __name__=='__main__':
    jobs=[(i,'lerf_'+scene,arm) for i,(scene,arm) in enumerate([('figurines','probability'),('figurines','joint'),('ramen','probability'),('ramen','joint')])]
    with ThreadPoolExecutor(4) as pool:rows=sum(pool.map(run,jobs),[])
    Path('output/subset_readout_next45/execution.json').write_text(json.dumps(rows,indent=2)+'\n')
