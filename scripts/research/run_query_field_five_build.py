"""Build each missing field with the same fixed recipe, without scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

from radio_gs.query_field.storage import sha256, field_identity, deployment_bytes
from scripts.research.prepare_query_field_five_benchmarks import ROOT, CONFIG, write


PY='/root/miniconda3/envs/cybersim_agent/bin/python'
SAM='/root/miniconda3/envs/sam3/bin/python'


def execute(command, log, environment):
    with Path(log).open('x') as handle:
        subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)


def validate_v2_closure_metadata(directory, store_root):
    from radio_gs.query_field.shared_storage import _safe_root, _blob_path, _read_manifest
    _,manifest=_read_manifest(directory)
    if manifest.get('blob_store') != 'explicit':
        raise ValueError('five-build requires an explicit shared store package')
    requested=_safe_root(store_root)
    for descriptor in manifest['tensors'].values():
        blob=_blob_path(requested,descriptor['sha256'])
        if not blob.is_file() or blob.stat().st_size != descriptor['bytes']:
            raise ValueError('cached blob closure is missing or has wrong size')
    return manifest, requested


def build(job, cohort, gpu, *, store_root=None, legacy_v1=False):
    if not legacy_v1 and store_root is None:
        raise ValueError("new five-build requires an explicit shared v2 store")
    if legacy_v1 and store_root is not None:
        raise ValueError("legacy v1 and shared store are mutually exclusive")
    root=Path(job['output']);started=time.monotonic()
    if (root/'build_complete.json').exists():
        existing=json.loads((root/'models/trained/manifest.json').read_text())
        expected_schema='radio_gs.query_preserving_field.v1' if legacy_v1 else 'radio_gs.query_preserving_field.v2'
        if existing['schema'] != expected_schema:
            raise ValueError('cached field format differs from explicit build mode; choose a fresh output')
        cached=json.loads((root/'build_complete.json').read_text())
        if not legacy_v1:
            existing,requested=validate_v2_closure_metadata(root/'models/trained',store_root)
            recorded=cached.get('blob_store')
            if not recorded or os.path.abspath(recorded) != str(requested):
                raise ValueError('cached blob store differs from requested store; choose a fresh output or explicitly validate relocation')
            if cached.get('field_schema') != expected_schema or cached.get('field_sha256') != field_identity(existing):
                raise ValueError('cached field identity differs from manifest')
        return cached
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
            command=[PY,'-m','radio_gs.query_field',stage,'--config',job['config'],'--output',str(root)]
            if stage == 'fit':
                command += ['--legacy-v1'] if legacy_v1 else ['--store-root',str(store_root)]
            execute(command,root/f'{stage}.log',env)
    exported=json.loads((root/'models/trained/manifest.json').read_text())
    expected_schema='radio_gs.query_preserving_field.v1' if legacy_v1 else 'radio_gs.query_preserving_field.v2'
    if exported['schema'] != expected_schema:
        raise ValueError('existing field format differs from explicit build mode; preserve it and choose a fresh output')
    if not legacy_v1:
        exported,_=validate_v2_closure_metadata(root/'models/trained',store_root)
    observations=json.loads((root/'observations/observations.json').read_text())
    report=dict(family=job['family'],scene=job['name'],gpu=gpu,seconds=time.monotonic()-started,
                field_sha256=field_identity(exported),field_schema=exported['schema'],elements=exported['elements'],
                observed=(exported['observed'] if 'observed' in exported else observations['observed_elements']),
                deployment_bytes=deployment_bytes(exported),
                blob_store=None if legacy_v1 else str(store_root),
                configuration_sha256=sha256(job['config']),observations_sha256=sha256(root/'observations/observations.json'),
                train_frames=job['train_frames'],development_frames=job['development_frames'],
                benchmark_labels_opened=False,geometry_condition=observations['geometry_condition'])
    write(root/'build_complete.json',report)
    print('built',job['family'],job['name'],'GPU',gpu,flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',nargs='+',type=int,default=[1,2,3,4,5])
    parser.add_argument('--family',choices=['nvos','scannet','uqis'])
    parser.add_argument('--store-root', type=Path, help='Required shared v2 blob store for new builds')
    parser.add_argument('--cohort', type=Path, help='Explicit cohort config; use fresh output paths for new builds')
    parser.add_argument('--summary-root', type=Path, help='New construction-summary destination')
    parser.add_argument('--legacy-v1', action='store_true', help='Explicit historical full-package rebuild')
    args=parser.parse_args(argv)
    if args.legacy_v1 and args.store_root is not None:
        parser.error('--legacy-v1 and --store-root are mutually exclusive')
    if not args.legacy_v1 and (args.store_root is None or args.cohort is None or args.summary_root is None):
        parser.error('new five-build requires --store-root, --cohort and --summary-root; no implicit v1 fallback')
    cohort_path=args.cohort if args.cohort is not None else CONFIG/'cohort.json'
    summary_root=args.summary_root if args.summary_root is not None else ROOT
    summary_path=summary_root/(f'{args.family}_construction_complete.json' if args.family else 'construction_complete.json')
    if summary_path.exists():raise FileExistsError(summary_path)
    cohort=json.loads(cohort_path.read_text())
    jobs=[j for j in cohort['construction'] if args.family is None or j['family']==args.family]
    # Interleave families to expose adapter integration problems early.
    jobs=sorted(jobs,key=lambda j:(j['name'],j['family']))
    lanes=[jobs[i::len(args.gpus)] for i in range(len(args.gpus))]
    def lane(index):
        return [build(job,cohort,args.gpus[index],store_root=args.store_root,legacy_v1=args.legacy_v1) for job in lanes[index]]
    with ThreadPoolExecutor(len(lanes)) as pool:
        results=[result for lane_result in pool.map(lane,range(len(lanes))) for result in lane_result]
    summary_root.mkdir(parents=True,exist_ok=True)
    write(summary_path,dict(jobs=results,cohort_sha256=sha256(cohort_path),
                           storage_mode='explicit_legacy_v1' if args.legacy_v1 else 'shared_v2',
                           blob_store=None if args.legacy_v1 else str(args.store_root)))


if __name__=='__main__':main()
