"""Fixed factorial: SigLIP / learned region map × outside-scope negative / unknown.

The two negative baselines are already complete. Here only the corresponding
unknown conditions run. Source positives/confidence, geometry and prior weight
stay unchanged. No coefficient or per-scene choice is introduced.
"""
import concurrent.futures,json,os,subprocess
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/region_negative_control');CONFIG=Path('configs/pipeline/region_negative_control')
SCENES=sorted(Path('configs/pipeline/scannet_identity_observations').glob('*.scene.json'))
GPUS=['0','1','2','3','4','5'];JOBS=[]
for mode in ['siglip','mask_adapter']:
    (CONFIG/mode).mkdir(parents=True,exist_ok=True)
    for path in SCENES:
        name=path.name.removesuffix('.scene.json')
        source=path if mode=='siglip' else Path('output/adapter_observation_control')/name/mode/'scene.json'
        scene=json.loads(source.read_text());scene['source_negative_policy']='unknown'
        dest=CONFIG/mode/path.name;dest.write_text(json.dumps(scene,indent=2)+'\n');JOBS.append((mode,dest))
def worker(i):
    for mode,path in JOBS[i::len(GPUS)]:
        scene=path.name.removesuffix('.scene.json');out=ROOT/mode/scene;out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=GPUS[i])
        with (out/'run.log').open('w') as log:
            subprocess.run([PYTHON,'scripts/research/run_scannet_query.py','--scene',str(path),'--query',f'configs/consolidated/scannet_native/{scene}.queries.json','--output',str(out/'query'),'--gpu',GPUS[i]],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([PYTHON,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'query/typed_readout.pt'),'--evaluation',f'configs/consolidated/scannet_native/{scene}.evaluation.json','--output',str(out/'evaluation'),'--point-only'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(mode,scene,'complete',flush=True)
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(len(GPUS)) as pool:list(pool.map(worker,range(len(GPUS))))
    rows={mode:{p.name.removesuffix('.scene.json'):json.loads((ROOT/mode/p.name.removesuffix('.scene.json')/'evaluation/report.json').read_text())['point_metrics'] for p in SCENES} for mode in ['siglip','mask_adapter']}
    macro={mode:{arm:{s:sum(r[arm][s]['miou'] for r in scenes.values())/len(scenes) for s in ['19','15','10']} for arm in ['map','source']} for mode,scenes in rows.items()}
    (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,independent_test=False,
        outside_scope_unknown=True,positive_and_visibility_mass_unchanged=True),indent=2)+'\n');print(macro)
