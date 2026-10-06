"""Temporal bridging and all-legal-source extent controls on fixed geometry."""
import json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
SAM='/root/miniconda3/envs/sam3/bin/python'
OUT=Path('output/typed_evidence_next44')

def run(job):
    gpu,scene=job;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
    tracks=OUT/'full_source_tracks'
    commands=[[SAM,'scripts/research/track_nvos_sources_next43.py','--scene',scene,'--all-legal-sources','--output',str(tracks)],
              [PYTHON,'scripts/research/fuse_typed_evidence_next44.py','--scene',scene,'--modes','bridge,full_video','--tracks-base',str(tracks)]]
    commands += [[PYTHON,'scripts/research/evaluate_nvos_scope_field.py','--scene',scene,'--base','output/taskwise_scope_composition_next38','--output-base',str(OUT/m),'--element-path',str(OUT/m/scene/'elements.pt')] for m in ['bridge','full_video']]
    rows=[]
    with (OUT/(scene+'_full_source.log')).open('w') as f:
        for c in commands:
            rc=subprocess.run(c,env=env,stdout=f,stderr=subprocess.STDOUT).returncode;rows.append(dict(scene=scene,command=c,exit_code=rc));print(scene,c[1],rc,flush=True)
            if rc:break
    return rows

if __name__=='__main__':
    scenes=['leaves','orchids','fern','flower','fortress','horns_center','horns_left','trex'];rows=[]
    for start in range(0,len(scenes),3):
        with ThreadPoolExecutor(3) as p:rows+=sum(p.map(run,[(3+i,s) for i,s in enumerate(scenes[start:start+3])]),[])
    (OUT/'full_source_execution.json').write_text(json.dumps(rows,indent=2)+'\n')
