"""Eight-query scale-matched source raster and longer-solver controls."""
import os,subprocess,json,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
SCENES=['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex']
ROOT=Path('output/reference_resolution');PY='/root/miniconda3/envs/cybersim_agent/bin/python'
parser=argparse.ArgumentParser();parser.add_argument('--arms',nargs='+',choices=['double_raster','long_solver','double_long'],default=['double_raster','long_solver','double_long']);args=parser.parse_args()
jobs=[(arm,s) for arm in args.arms for s in SCENES]
def worker(gpu):
    for arm,s in jobs[gpu::6]:
        out=ROOT/arm/s;out.mkdir(parents=True,exist_ok=True);base=f'configs/consolidated/nvos/{s}';scene=json.loads(Path(base+'.scene.json').read_text())
        if arm in ['double_raster','double_long']:scene['source_fit_hw']=[v*2 for v in scene['source_fit_hw']];scene['boundary_radius']*=2
        config=out/'scene.json';config.write_text(json.dumps(scene,indent=2)+'\n')
        commands=[[PY,'scripts/research/fit_reference_instance.py','--scene',str(config),'--query',base+'.query.json','--tracks',f'output/reference_integration/fresh_tracks/{s}/tracks.pt','--steps','120' if arm=='double_raster' else '480','--output',str(out)],[PY,'scripts/research/evaluate_reference_instance.py','--scene',str(config),'--evaluation',base+'.evaluation.json','--field',str(out/'field.pt'),'--output',str(out/'evaluation')]]
        if arm=='double_raster':commands.append([PY,'scripts/research/diagnose_reference_conflict.py','--scene',s,'--scene-input',str(config),'--output',str(out/'conflict')])
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        with (out/'run.log').open('w') as log:
            for cmd in commands:subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
        print(arm,s,'done',flush=True)
with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(worker,range(6)))
