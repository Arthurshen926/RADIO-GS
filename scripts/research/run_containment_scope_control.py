"""Full four-scene, two-head containment controls. No query-specific routes."""
import os,subprocess,json,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
parser=argparse.ArgumentParser();parser.add_argument('--scope-assignment',action='store_true');args=parser.parse_args()
PY='/root/miniconda3/envs/cybersim_agent/bin/python';scenes=['figurines','ramen','teatime','waldo_kitchen'];root=Path('output/containment_scope_assignment' if args.scope_assignment else 'output/containment_scope_control')
def worker(job):
    gpu,s=job;out=root/s;out.mkdir(parents=True,exist_ok=True);base=Path('output/conditioned_identity/full_source')/s;commands=[[PY,'scripts/research/build_containment_unknown_evidence.py','--inputs',str(base/'identity_evidence/inputs.json'),'--output',str(out/'evidence')]]
    if args.scope_assignment:commands[0].append('--scope-assignment')
    for arm in ['assignment'] if args.scope_assignment else ['structural','semantic']:
        dest=out/arm;commands.extend([[PY,'scripts/research/decode_expanded_extent_next72.py','--inputs',str(out/'evidence'/arm/'inputs.json'),'--key','lerf_'+s,'--output-base',str(dest/'sets'),'--mode','admission','--scope-initialization','source','--pixel-loss','balanced','--support-prior','footprint','--raster-scale','2','--steps','240'],[PY,'scripts/research/fit_consolidated_response.py','--inputs',str(out/'evidence'/arm/'inputs.json'),'--readout',str(dest/'sets'/('lerf_'+s)/'typed_readout.pt'),'--output',str(dest/'response'),'--policies','native_protected','--solver','backprojection'],[PY,'scripts/research/evaluate_consolidated_query.py','--scene',f'configs/consolidated/{s}.scene.json','--readout',str(dest/'response/native_protected/typed_readout.pt'),'--output',str(dest/'evaluation')]])
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (out/'run.log').open('w') as log:
        for command in commands:subprocess.run(command,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
    print(s,'complete',flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,enumerate(scenes)))
