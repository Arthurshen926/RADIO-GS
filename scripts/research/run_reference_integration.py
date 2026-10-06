"""Execute explicit reference-instance replay or fresh source tracking and fit."""
import argparse,json,os,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PY='/root/miniconda3/envs/cybersim_agent/bin/python';SAM='/root/miniconda3/envs/sam3/bin/python'
SCENES=['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex']


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=['replay','fresh','subset'],required=True);p.add_argument('--gpus',nargs='+',default=['0','1','2','3','4','5']);a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True);jobs=[]
    for scene in SCENES:
        dest=root/a.mode/scene;dest.mkdir(parents=True,exist_ok=True);config=f'configs/consolidated/nvos/{scene}.scene.json';query=f'configs/consolidated/nvos/{scene}.query.json';evaluation=f'configs/consolidated/nvos/{scene}.evaluation.json';commands=[]
        tracks=f'output/typed_evidence_next44/full_source_tracks/{scene}/tracks.pt'
        if a.mode=='fresh':
            commands.append([SAM,'scripts/research/track_nvos_sources_next43.py','--scene',scene,'--scene-input',config,'--query',query,'--all-legal-sources','--output',str(root/'fresh_tracks')]);tracks=str(root/'fresh_tracks'/scene/'tracks.pt')
        if a.mode=='subset':
            commands.append([PY,'scripts/research/decode_expanded_extent_next72.py','--inputs',str(root/'fresh'/scene/'subset_inputs.json'),'--key',scene,'--output-base',str(dest/'sets'),'--mode','admission','--scope-initialization','source','--pixel-loss','balanced','--support-prior','footprint','--steps','240','--boundary-radius','2'])
            field=dest/'sets'/scene/'typed_readout.pt'
        else:
            commands.append([PY,'scripts/research/fit_reference_instance.py','--scene',config,'--query',query,'--tracks',tracks,'--output',str(dest)]);field=dest/'field.pt'
        commands.append([PY,'scripts/research/evaluate_reference_instance.py','--scene',config,'--evaluation',evaluation,'--field',str(field),'--output',str(dest/'evaluation')]+(['--subset-field'] if a.mode=='subset' else []))
        jobs.append(dict(scene=scene,directory=str(dest),commands=commands))
    (root/(a.mode+'_plan.json')).write_text(json.dumps(jobs,indent=2)+'\n')
    def worker(gpu,jobs):
        for job in jobs:
            env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4')
            with (Path(job['directory'])/'run.log').open('w') as log:
                for c in job['commands']:subprocess.run(c,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(a.mode,job['scene'],'complete',flush=True)
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        fs=[pool.submit(worker,gpu,jobs[i::len(a.gpus)]) for i,gpu in enumerate(a.gpus)]
        for f in fs:f.result()


if __name__=='__main__':main()
