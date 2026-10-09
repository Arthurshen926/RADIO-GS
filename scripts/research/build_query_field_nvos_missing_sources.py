"""Complete the sealed123-frame NVOS query-free cache preparation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from scripts.research.prepare_query_field_nvos_missing_sources import ROOT
from scripts.research.run_query_field_relations import read,write,receipt,check


def build(row,gpu):
    check(row['job']);job=read(row['job']['path']);root=ROOT/row['scene']
    for e in job['frames']:check(dict(path=e['rgb'],sha256=e['rgb_sha256']))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='.',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    with (root/'sam.log').open('x') as handle:
        subprocess.run(job['sam_command'],env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    raw=read(root/'source_sam/manifest.json');by_frame={int(Path(r['image']).stem.split('_')[-1]):r for r in raw['images']}
    entries=[]
    for e in job['frames']:
        sam=by_frame[e['frame']]
        if sam['source_image_sha256']!=e['rgb_sha256']:raise ValueError('new source SAM RGB differs')
        check(dict(path=sam['output'],sha256=sam['output_sha256']))
        entries.append(dict(frame=e['frame'],rgb=e['rgb'],rgb_sha256=e['rgb_sha256'],sam=sam['output'],sam_sha256=sam['output_sha256']))
    cfg=root/'native_config.json';write(cfg,dict(output=str(root/'native_regions'),encoder=job['encoder'],frames=entries))
    with (root/'native.log').open('x') as handle:
        subprocess.run([sys.executable,'-m','radio_gs.query_field.source_regions','--config',str(cfg)],env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    encoded=read(root/'native_regions/encoding.json')
    if [e['frame'] for e in encoded['frames']]!=[e['frame'] for e in entries]:raise ValueError('new native coverage differs')
    for e in encoded['frames']:check(dict(path=str(root/'native_regions'/f"frame_{e['frame']:05d}.pt"),sha256=e['output_sha256']))
    report=dict(scene=row['scene'],frames=len(entries),regions=sum(e['regions'] for e in encoded['frames']),
        sam=receipt(root/'source_sam/manifest.json'),native=receipt(root/'native_regions/encoding.json'),
        full_legal_frames=job['full_legal_frames'],excluded_frames=job['excluded_development'],
        native_directories=[*job['existing_native_directories'],str(root/'native_regions')],
        query_free=True,queries_read=False,labels_read=False,field_transfer_complete=False)
    write(root/'cache_complete.json',report);print('native complete',row['scene'],len(entries),flush=True)
    return report


def main(gpus):
    plan=read(ROOT/'preparation.json');jobs=plan['jobs'];chunks=[jobs[i::len(gpus)] for i in range(len(gpus))]
    def lane(i):return [build(row,gpus[i]) for row in chunks[i]]
    with ThreadPoolExecutor(len(gpus)) as pool:reports=[r for batch in pool.map(lane,range(len(gpus))) for r in batch]
    if sum(r['frames'] for r in reports)!=123:raise ValueError('incomplete new caches')
    write(ROOT/'cache_complete.json',dict(scenes=7,frames=123,regions=sum(r['regions'] for r in reports),records=reports,
        preparation=receipt(ROOT/'preparation.json'),code=receipt(Path(__file__)),field_transfer_complete=False,
        benchmark_queries_evaluated=False,no_new_benchmark_performance_claim=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='2,3');a=p.parse_args();main(list(map(int,a.gpus.split(','))))
