"""Two fixed reliability mechanisms, same v3 assets/source fusion/evaluation."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess

PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/observation_reliability')
CONFIG=Path('configs/pipeline/observation_reliability')
SCENES=sorted(Path('configs/pipeline/scannet_observation_complete').glob('*.scene.json'))
ARMS=['entropy_information','heldout_view_support']


def worker(gpu):
    import torch
    for path in SCENES[gpu::6]:
        name=path.name.removesuffix('.scene.json');out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
        scene=json.loads(path.read_text())
        # Offsets are source provenance, not a changed descriptor or geometry.
        original=json.loads(Path('configs/pipeline/scannet_native_complete',path.name).read_text())
        field=torch.load(scene['native_observation_field'],map_location='cpu',weights_only=False)
        offsets=[0]
        for frame in original['source_frames']:
            row=torch.load(Path(original['native_region_directories'][0])/f'frame_{frame:05d}.pt',map_location='cpu',weights_only=False)
            offsets.append(offsets[-1]+len(row['descriptors']))
        field['region_offsets']=torch.tensor(offsets)
        field['source_frames']=original['source_frames']
        torch.save(field,out/'observation_field.pt');del field
        for arm in ARMS:
            candidate=dict(scene,native_observation_field=str(out/'observation_field.pt'),native_observation_reliability=arm)
            cfg=CONFIG/arm/path.name;cfg.parent.mkdir(parents=True,exist_ok=True);cfg.write_text(json.dumps(candidate,indent=2)+'\n')
            dest=out/arm;dest.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu))
            with (dest/'run.log').open('w') as log:
                subprocess.run([PYTHON,'-m','radio_gs.pipeline','--scene',str(cfg),'--queries',f'configs/consolidated/scannet_native/{name}.queries.json','--access','source-assisted','--output',str(dest/'query'),'--gpu',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([PYTHON,'scripts/research/evaluate_scannet_runtime.py','--readout',str(dest/'query/category/typed_readout.pt'),'--evaluation',f'configs/consolidated/scannet_native/{name}.evaluation.json','--output',str(dest/'evaluation'),'--point-only'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(name,arm,'complete',flush=True)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(worker,range(6)))
    rows={}
    for path in SCENES:
        n=path.name.removesuffix('.scene.json')
        rows[n]={'v3':json.loads(Path('output/system_v3/scannet',n,'evaluation/report.json').read_text())['point_metrics']}
        for arm in ARMS:rows[n][arm]=json.loads((ROOT/n/arm/'evaluation/report.json').read_text())['point_metrics']
    macro={arm:{a:{s:sum(r[arm][a][s]['miou'] for r in rows.values())/len(rows) for s in ['19','15','10']} for a in ['map','source']} for arm in ['v3',*ARMS]}
    (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,independent_test=False,new_teacher_calls=0,parameter_sweep=False),indent=2)+'\n')
    print(macro,flush=True)
