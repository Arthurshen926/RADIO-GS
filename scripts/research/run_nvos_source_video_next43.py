"""Restricted-source video pilot and fixed existing NVOS evaluation."""
import json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def run(job):
    gpu,scene=job;out=Path('output/source_video_next43');out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
    commands=[('sam3',['scripts/research/track_nvos_sources_next43.py','--scene',scene]),('cybersim_agent',['scripts/research/fuse_nvos_video_next43.py','--scene',scene])]
    commands += [('cybersim_agent',['scripts/research/evaluate_nvos_scope_field.py','--scene',scene,'--base','output/taskwise_scope_composition_next38','--output-base',str(out/mode),'--element-path',str(out/mode/scene/'elements.pt')]) for mode in ['positive','signed']]
    results=[]
    with (out/(scene+'.log')).open('w') as log:
        for python,cmd in commands:
            rc=subprocess.run([f'/root/miniconda3/envs/{python}/bin/python']+cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            results.append(dict(scene=scene,command=cmd,exit_code=rc));print(scene,cmd[0],rc,flush=True)
            if rc:break
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--scenes',default='leaves,orchids');args=parser.parse_args()
    scenes=args.scenes.split(',');rows=[]
    for start in range(0,len(scenes),3):
        jobs=[(3+i,scene) for i,scene in enumerate(scenes[start:start+3])]
        with ThreadPoolExecutor(len(jobs)) as pool:rows+=sum(pool.map(run,jobs),[])
    Path('output/source_video_next43/execution_'+('_'.join(scenes))+'.json').write_text(json.dumps(rows,indent=2)+'\n')
