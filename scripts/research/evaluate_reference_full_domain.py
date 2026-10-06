"""Evaluate the predeclared full-mask shared field; single-view fits never deploy."""
import os,subprocess,json,torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('output/reference_representability');SCENES=['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex']
def worker(gpu):
    for s in SCENES[gpu::6]:
        out=ROOT/s;field=torch.load(out/'joint_full.pt',map_location='cpu',weights_only=False);baseline=torch.load(f'output/reference_integration/fresh/{s}/field.pt',map_location='cpu',weights_only=False)
        # Full-mask joint fit uses all visible source pixels. Keep the same
        # deployment observed-support contract as the frozen baseline.
        field['observed']=baseline['observed'];field['elements'][~field['observed']]=0;torch.save(field,out/'full_domain_field.pt')
        with (out/'evaluation.log').open('w') as log:subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/research/evaluate_reference_instance.py','--scene',f'configs/consolidated/nvos/{s}.scene.json','--evaluation',f'configs/consolidated/nvos/{s}.evaluation.json','--field',str(out/'full_domain_field.pt'),'--output',str(out/'full_domain_evaluation')],env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
        print(s,'full-domain evaluated',flush=True)
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(worker,range(6)))
