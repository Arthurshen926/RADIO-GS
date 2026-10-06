"""Fixed two-arm text pilots or complete reference cohort; no scene routing."""
import argparse,json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'


def run(job):
    gpu,name,reference,modes=job;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
    out=Path('output/typed_evidence_next44');out.mkdir(parents=True,exist_ok=True)
    commands=[[PYTHON,'scripts/research/fuse_typed_evidence_next44.py','--scene' if reference else '--key',name,'--modes',','.join(modes)]]
    if reference:
        commands += [[PYTHON,'scripts/research/evaluate_nvos_scope_field.py','--scene',name,'--base','output/taskwise_scope_composition_next38','--output-base',str(out/m),'--element-path',str(out/m/name/'elements.pt')] for m in modes]
    else:
        commands += [[PYTHON,'scripts/research/evaluate_scope_dictionary.py','--key',name,'--base',str(out/m),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/m/name/'elements.pt')] for m in modes]
    rows=[]
    with (out/(name+'_run.log')).open('w') as f:
        for command in commands:
            rc=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT).returncode;rows.append(dict(name=name,command=command,exit_code=rc));print(name,command[1],rc,flush=True)
            if rc:break
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',action='store_true');p.add_argument('--modes');p.add_argument('--names',required=True);p.add_argument('--gpus',default='3,4,5');a=p.parse_args()
    modes=a.modes.split(',') if a.modes else (['corroborated'] if a.reference else ['direct','corroborated']);names=a.names.split(',');gpus=list(map(int,a.gpus.split(',')));rows=[]
    for start in range(0,len(names),len(gpus)):
        with ThreadPoolExecutor(len(gpus)) as pool:rows+=sum(pool.map(run,[(gpus[i],n,a.reference,modes) for i,n in enumerate(names[start:start+len(gpus)])]),[])
    (Path('output/typed_evidence_next44')/('execution_'+('_'.join(modes))+'_'+('_'.join(names))+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
