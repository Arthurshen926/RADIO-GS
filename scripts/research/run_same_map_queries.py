"""Bounded same-map image/3D/2D prompt integration; not a new benchmark protocol."""
import json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PY='/root/miniconda3/envs/cybersim_agent/bin/python';ROOT=Path('output/same_map_queries')


def run(item):
    probe,mode,gpu=item;dest=ROOT/f'probe_{probe}_{mode}';dest.mkdir(parents=True,exist_ok=True);scene=str(ROOT/'probes/scene.json');query=str(ROOT/'probes'/f'probe_{probe}_{mode}.json');env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4')
    commands=[[PY,'scripts/research/resolve_map_query.py','--scene',scene,'--query',query,'--output',str(dest/'resolved')],[PY,'scripts/research/run_reference_query.py','--scene',scene,'--query',str(dest/'resolved/query.json'),'--output',str(dest/'runtime'),'--gpu',str(gpu)]]
    if mode=='image_masked':
        commands=[[PY,'scripts/research/run_reference_query.py','--scene',scene,'--query',query,'--output',str(dest/'runtime'),'--gpu',str(gpu)]]
    (dest/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (dest/'run.log').open('w') as log:
        for c in commands:subprocess.run(c,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(probe,mode,'done',flush=True)


def main():
    # 2D prompts are the exact visibility-resolved form of 3D prompts, a coordinate
    # equivalence control rather than an independent annotation/accuracy test.
    first=[(p,m,2*p+i) for p in range(2) for i,m in enumerate(['image','point3d'])]
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,first))
    jobs=[]
    for p in range(2):
        q=json.loads((ROOT/f'probe_{p}_point3d/resolved/query.json').read_text());q['modality']='prompt_2d';q['id']=f'probe_{p}_point2d';q.pop('world_xyz',None);q.pop('geometry',None)
        (ROOT/'probes'/f'probe_{p}_point2d.json').write_text(json.dumps(q,indent=2)+'\n');jobs.append((p,'point2d',4+p))
    jobs.extend((p,'image_masked',p) for p in range(2))
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,jobs))


if __name__=='__main__':main()
