"""Paired query-independent maps and full frozen evaluation on all ten scenes."""
import concurrent.futures,json,os,subprocess
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/adapter_observation_control')
SCENES=sorted(Path('configs/pipeline/scannet_identity_observations').glob('*.scene.json'))
GPUS=['0','1','2','3','4','5']
def worker(i):
    for path in SCENES[i::len(GPUS)]:
        scene=path.name.removesuffix('.scene.json');out=ROOT/scene;out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=GPUS[i])
        with (out/'run.log').open('w') as log:
            subprocess.run([PYTHON,'-m','radio_gs.pipeline.adapter_observation_field','--scene',str(path),
                '--regions',f'output/scannet_native_completion/{scene}/native_regions','--output',str(out)],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            for mode in ['mask_pool','mask_adapter']:
                dest=out/mode
                subprocess.run([PYTHON,'scripts/research/run_scannet_query.py','--scene',str(dest/'scene.json'),
                    '--query',f'configs/consolidated/scannet_native/{scene}.queries.json','--output',str(dest/'query'),
                    '--gpu',GPUS[i]],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([PYTHON,'scripts/research/evaluate_scannet_runtime.py','--readout',str(dest/'query/typed_readout.pt'),
                    '--evaluation',f'configs/consolidated/scannet_native/{scene}.evaluation.json',
                    '--output',str(dest/'evaluation'),'--point-only'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                print(scene,mode,'complete',flush=True)
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(len(GPUS)) as pool:list(pool.map(worker,range(len(GPUS))))
    rows={mode:{p.name.removesuffix('.scene.json'):json.loads((ROOT/p.name.removesuffix('.scene.json')/mode/'evaluation/report.json').read_text())['point_metrics'] for p in SCENES} for mode in ['mask_pool','mask_adapter']}
    macro={mode:{arm:{s:sum(r[arm][s]['miou'] for r in scenes.values())/len(scenes) for s in ['19','15','10']} for arm in ['map','source']} for mode,scenes in rows.items()}
    (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,independent_test=False,
        source_solver='unchanged v5',query_independent_map_build=True,geometry_and_transfer_unchanged=True),indent=2)+'\n');print(macro)
