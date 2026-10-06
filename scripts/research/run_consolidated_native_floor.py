"""One fixed native-response lower-bound control, never query-wise selection."""
import json,torch,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
scenes=['figurines','ramen','teatime','waldo_kitchen']
def worker(job):
    gpu,s=job;root=Path('output/consolidated_response_control')/s;d=torch.load(root/'fusion/native_protected/typed_readout.pt',map_location='cpu',weights_only=False);base=torch.load(Path('output/conditioned_identity/full_source')/s/'sets'/('lerf_'+s)/'typed_readout.pt',map_location='cpu',weights_only=False);out=root/'native_floor';out.mkdir(exist_ok=True);d['scores']=torch.maximum(d['scores'],base['scores']);d['response_policy']='fixed pointwise max(native,response fusion); not calibrated truth';torch.save(d,out/'typed_readout.pt')
    with (out/'run.log').open('w') as log:subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/research/evaluate_consolidated_query.py','--scene',f'configs/consolidated/{s}.scene.json','--readout',str(out/'typed_readout.pt'),'--output',str(out/'evaluation')],env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
    print(s,'native-floor evaluated',flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,enumerate(scenes)))
