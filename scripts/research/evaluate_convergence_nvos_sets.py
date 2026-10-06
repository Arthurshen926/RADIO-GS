"""Supplement retained NVOS response scores with actual extracted-set projections."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=['v8','local','partial','ownership'],required=True)
    parser.add_argument('--gpu',default='0');args=parser.parse_args()
    root=Path('output/evidence_convergence_20261005')
    path=Path('output/system_v8_membership_closure_20261005') if args.condition=='v8' else root/(args.condition+'_system')
    summary=json.loads((path/'summary.json').read_text());rows=[]
    for result in summary['rows']:
        job=result['job']
        if job['benchmark']!='nvos':continue
        out=root/'nvos_sets'/args.condition/job['name'];out.mkdir(parents=True,exist_ok=True)
        command=['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/research/evaluate_reference_instance.py',
            '--scene',job['scene'],'--evaluation',job['evaluation'],
            '--field',str(path/'nvos'/job['name']/'query/typed_readout.pt'),
            '--selected-subset','--output',str(out)]
        with (out/'evaluate.log').open('w') as log:
            subprocess.run(command,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=args.gpu,OPENBLAS_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        measured=json.loads((out/'evaluation.json').read_text())
        rows.append(dict(scene=job['name'],response_iou=result['metrics']['iou'],selected_projection_iou=measured['metrics']['iou']))
    report=dict(condition=args.condition,rows=rows,selected_projection_iou=sum(r['selected_projection_iou'] for r in rows)/len(rows),
        additional_diagnostic=True,retained_headline_unchanged=True,
        independent_3d_ground_truth=False,output='actual selected Gaussian subset; alpha quantized to 8-bit and >10')
    (root/'nvos_sets'/(args.condition+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':main()
