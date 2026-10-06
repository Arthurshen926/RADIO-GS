"""Replay the frozen eight-query positive-only cohort with a global extent cut."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='2,3,4,5');p.add_argument('--strength',type=float,default=1.);p.add_argument('--geometry',choices=['local','retained'],default='local');p.add_argument('--evidence',choices=['learned','scopes'],default='learned');p.add_argument('--readout',choices=['graph_cut','reference_scopes'],default='graph_cut');a=p.parse_args()
    tag=str(a.strength) if (a.geometry,a.evidence)==('local','learned') else f'{a.geometry}_{a.evidence}_{a.strength}'
    if a.readout=='reference_scopes':tag=a.readout
    out=Path('output/seeded_extent_cut_next40')/tag;out.mkdir(parents=True,exist_ok=True)
    jobs=queue.Queue()
    for scene in ['fern','flower','fortress','horns_center','horns_left','leaves','orchids','trex']:jobs.put(scene)
    def run(gpu):
        results=[]
        while not jobs.empty():
            try:scene=jobs.get_nowait()
            except queue.Empty:break
            command=['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/research/evaluate_nvos_scope_field.py','--scene',scene,'--base','output/taskwise_scope_composition_next38','--prompt-readout',a.readout,'--graph-strength',str(a.strength),'--graph-geometry',a.geometry,'--graph-evidence',a.evidence,'--output-base',str(out)]
            with (out/(scene+'.log')).open('w') as log:
                rc=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT).returncode
            row=dict(scene=scene,exit_code=rc);results.append(row);print(row,flush=True)
        return results
    with ThreadPoolExecutor(len(a.gpus.split(','))) as pool:results=sum(pool.map(run,a.gpus.split(',')),[])
    (out/'execution.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
