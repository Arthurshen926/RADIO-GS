import os,json,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PY='/root/miniconda3/envs/cybersim_agent/bin/python';SAM='/root/miniconda3/envs/sam3/bin/python';OUT=Path('output/subset_readout_next45')
def run(job):
    gpu,key=job;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
    commands=[[SAM,'scripts/research/detect_typed_sources_next44.py','--key',key,'--output-base',str(OUT/'highres_source'),'--raster-scale','2'],[PY,'scripts/research/validate_typed_identity_next44.py','--key',key,'--evidence-base',str(OUT/'highres_source')],[PY,'scripts/research/fuse_highres_evidence_next45.py','--key',key],[PY,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(OUT/'highres'),'--readout','anchored','--pipeline','--calibrated','--element-results',str(OUT/'highres'/key/'elements.pt')]];rows=[]
    with (OUT/(key+'_highres.log')).open('w') as f:
        for c in commands:
            rc=subprocess.run(c,env=env,stdout=f,stderr=subprocess.STDOUT).returncode;rows.append(dict(key=key,command=c,exit_code=rc));print(key,c[1],rc,flush=True)
            if rc:break
    return rows
if __name__=='__main__':
    with ThreadPoolExecutor(4) as pool:rows=sum(pool.map(run,enumerate(['lerf_figurines','lerf_ramen','lerf_teatime','lerf_waldo_kitchen'])),[])
    (OUT/'highres_execution.json').write_text(json.dumps(rows,indent=2)+'\n')
