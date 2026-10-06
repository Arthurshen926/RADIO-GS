"""Complete fixed soft-identity recipe across LERF4 and ScanNet8."""
import json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
SAM='/root/miniconda3/envs/sam3/bin/python'
OUT=Path('output/typed_evidence_next44')

def run(job):
    gpu,key=job;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
    commands=[[SAM,'scripts/research/detect_typed_sources_next44.py','--key',key],
              [PYTHON,'scripts/research/validate_typed_identity_next44.py','--key',key],
              [PYTHON,'scripts/research/fuse_typed_evidence_next44.py','--key',key,'--modes','soft_identity'],
              [PYTHON,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(OUT/'soft_identity'),'--readout','anchored','--pipeline','--calibrated','--element-results',str(OUT/'soft_identity'/key/'elements.pt')]]
    rows=[]
    with (OUT/(key+'_full_text.log')).open('w') as f:
        for c in commands:
            rc=subprocess.run(c,env=env,stdout=f,stderr=subprocess.STDOUT).returncode;rows.append(dict(key=key,command=c,exit_code=rc));print(key,c[1],rc,flush=True)
            if rc:break
    return rows

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,2');p.add_argument('--keys');a=p.parse_args();gpus=list(map(int,a.gpus.split(',')))
    keys=a.keys.split(',') if a.keys else [p.parents[2].name for p in sorted(Path('output/taskwise_scope_composition_next38').glob('*/anchored/pipeline_calibrated/evaluation.json')) if p.parents[2].name.startswith(('lerf_','scannet_'))]
    keys=[k for k in keys if not (OUT/'soft_identity'/k/'anchored/pipeline_calibrated/evaluation.json').exists()];rows=[]
    for start in range(0,len(keys),len(gpus)):
        with ThreadPoolExecutor(len(gpus)) as p:rows+=sum(p.map(run,[(gpus[i],k) for i,k in enumerate(keys[start:start+len(gpus)])]),[])
    (OUT/('full_text_execution_'+('_'.join(map(str,gpus)))+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
