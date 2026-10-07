"""Build each missing field with the same fixed recipe, without scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

from radio_gs.query_field.storage import sha256
from scripts.research.prepare_query_field_five_benchmarks import ROOT, CONFIG, write


PY='/root/miniconda3/envs/cybersim_agent/bin/python'
SAM='/root/miniconda3/envs/sam3/bin/python'


def execute(command, log, environment):
    with Path(log).open('x') as handle:
        subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)


def build(job, cohort, gpu):
    root=Path(job['output']);started=time.monotonic()
    if (root/'build_complete.json').exists():
        return json.loads((root/'build_complete.json').read_text())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
    if job['needs_sam']:
        if not (root/'source_sam/manifest.json').exists():
            execute([SAM,'-m','radio_gs.scripts.build_sam3_automatic_mask_cache',
                     '--image-root',str(root/'rgb'),'--output-root',str(root/'source_sam'),
                     '--checkpoint-path',cohort['sam_checkpoint'],'--grid-size','12',
                     '--minimum-area-fraction','0.0001','--maximum-area-fraction','0.9'],root/'sam.log',env)
        if not (root/'native_regions/encoding.json').exists():
            manifest=json.loads((root/'source_sam/manifest.json').read_text())
            by_stem={Path(r['image']).stem:r for r in manifest['images']}
            region_config=root/'regions_config.json'
            if not region_config.exists():
                entries=[]
                for image in job['images']:
                    row=by_stem[Path(image['rgb']).stem]
                    entries.append(dict(**image,sam=row['output'],sam_sha256=row['output_sha256']))
                write(region_config,dict(output=str(root/'native_regions'),encoder=cohort['encoder'],frames=entries))
            execute([PY,'-m','radio_gs.query_field.source_regions','--config',str(region_config)],root/'regions.log',env)
    for stage,receipt in [('compile',root/'observations/observations.json'),('fit',root/'models/trained/manifest.json')]:
        if not receipt.exists():
            execute([PY,'-m','radio_gs.query_field',stage,'--config',job['config'],'--output',str(root)],root/f'{stage}.log',env)
    exported=json.loads((root/'models/trained/manifest.json').read_text())
    observations=json.loads((root/'observations/observations.json').read_text())
    report=dict(family=job['family'],scene=job['name'],gpu=gpu,seconds=time.monotonic()-started,
                field_sha256=exported['files']['field.pt']['sha256'],elements=exported['elements'],
                observed=exported['observed'],deployment_bytes=exported['files']['field.pt']['bytes'],
                configuration_sha256=sha256(job['config']),observations_sha256=sha256(root/'observations/observations.json'),
                train_frames=job['train_frames'],development_frames=job['development_frames'],
                benchmark_labels_opened=False,geometry_condition=observations['geometry_condition'])
    write(root/'build_complete.json',report)
    print('built',job['family'],job['name'],'GPU',gpu,flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',nargs='+',type=int,default=[1,2,3,4,5])
    parser.add_argument('--family',choices=['nvos','scannet','uqis'])
    args=parser.parse_args();cohort=json.loads((CONFIG/'cohort.json').read_text())
    jobs=[j for j in cohort['construction'] if args.family is None or j['family']==args.family]
    # Interleave families to expose adapter integration problems early.
    jobs=sorted(jobs,key=lambda j:(j['name'],j['family']))
    lanes=[jobs[i::len(args.gpus)] for i in range(len(args.gpus))]
    def lane(index):
        return [build(job,cohort,args.gpus[index]) for job in lanes[index]]
    with ThreadPoolExecutor(len(lanes)) as pool:
        results=[result for lane_result in pool.map(lane,range(len(lanes))) for result in lane_result]
    write(ROOT/(f'{args.family}_construction_complete.json' if args.family else 'construction_complete.json'),dict(jobs=results))


if __name__=='__main__':main()
