"""Complete public-entry evaluation of the compiled native observation field."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess

PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/compiled_native_observations')
SCENES=sorted(Path('configs/pipeline/scannet_native_complete').glob('*.scene.json'))
GPUS=['1','5']
READOUT_ONLY=False

def worker(index):
    gpu=GPUS[index]
    for path in SCENES[index::len(GPUS)]:
        name=path.name.removesuffix('.scene.json');out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpu)
        with (out/'run.log').open('w') as log:
            def run(*argv):subprocess.run([PYTHON,*argv],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if not READOUT_ONLY:
                run('scripts/research/compile_native_observations.py','--scene',str(path),'--output',str(out))
            scene=json.loads((out/'scene.json').read_text())
            for key in ['native_field','native_region_directories','source_cameras']:
                scene.pop(key,None)
            scene['map_contract']['representation']='native_region_dictionary_with_sparse_element_weights'
            (out/'scene.json').write_text(json.dumps(scene,indent=2)+'\n')
            run('-m','radio_gs.pipeline','--scene',str(out/'scene.json'),'--queries',f'configs/consolidated/scannet_native/{name}.queries.json','--access','source-assisted','--output',str(out/'query'),'--gpu',gpu)
            run('scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'query/category/typed_readout.pt'),'--evaluation',f'configs/consolidated/scannet_native/{name}.evaluation.json','--output',str(out/'evaluation'),'--point-only')
        print(name,'complete',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--readout-only',action='store_true')
    READOUT_ONLY=parser.parse_args().readout_only
    with concurrent.futures.ThreadPoolExecutor(len(GPUS)) as pool:list(pool.map(worker,range(len(GPUS))))
    rows={p.name.removesuffix('.scene.json'):json.loads((ROOT/p.name.removesuffix('.scene.json')/'evaluation/report.json').read_text())['point_metrics'] for p in SCENES}
    reference=json.loads(Path('output/native_observation_public/summary.json').read_text())['scenes']
    difference=max(abs(rows[n][a][s]['miou']-reference[n][a][s]['miou']) for n in rows for a in ['map','source'] for s in ['19','15','10'])
    macro={a:{s:sum(r[a][s]['miou'] for r in rows.values())/len(rows) for s in ['19','15','10']} for a in ['map','source']}
    costs={n:dict(build=json.loads((ROOT/n/'build.json').read_text()),query=json.loads((ROOT/n/'query/category/result.json').read_text())['scope_decision']['observations']) for n in rows}
    (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,costs=costs,maximum_metric_difference=difference,independent_test=False),indent=2)+'\n')
    print(macro,difference,flush=True)
