"""One information intervention, two fixed identity controls, full UQIS cohorts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

PY='/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('output/identity_bottleneck_20261005'))
    p.add_argument('--gpus',default='0,1,2,4,5');p.add_argument('--stage',choices=['baseline','build','run'],required=True)
    args=p.parse_args();root=args.root;gpus=args.gpus.split(',')
    recipe=json.loads(Path('configs/pipeline/system_v8.json').read_text())
    jobs=[j for j in recipe['jobs'] if j['benchmark']=='uqis']
    if args.stage=='baseline':
        for job in jobs:
            source=Path(job['scene']);spec=json.loads(source.read_text());dest=root/'maps'/job['name']
            dest.mkdir(parents=True,exist_ok=True)
            for name in spec['map_files']:
                if not (dest/name).exists():os.link(source.parent/name,dest/name)
            spec['save_identity_evidence']=True
            (dest/'map.json').write_text(json.dumps(spec,indent=2)+'\n');job['scene']=str(dest/'map.json')
        recipe.update(default=False,name='Identity seed counterfactuals with unchanged v8 inference',
            comparison='v8 numeric rules unchanged; export region identity scores for a separate post-prediction diagnosis',
            validation=dict(status='v8 numerical control; this invocation runs nine UQIS rooms only',public_tasks_rerun=False))
        path=root/'baseline_recipe.json';path.write_text(json.dumps(recipe,indent=2)+'\n')
        subprocess.run([PY,'-u','-m','radio_gs.pipeline.cohort','--recipe',str(path),
            '--output',str(root/'baseline_system'),'--gpus',args.gpus,'--benchmarks','uqis'],check=True)
        return
    def lane(index):
        for job in jobs[index::len(gpus)]:
            dest=root/'context_maps'/job['name'];dest.mkdir(parents=True,exist_ok=True)
            if not (dest/'build.json').exists():
                command=[PY,'-u','-m','radio_gs.pipeline.regional_context_map','--scene',job['scene'],
                    '--mapping','output/scannet_uqis_9_v0_3_candidate/release_final/mapping/scenes.json',
                    '--masks-root','output/uqis_map_only_v1/maps/'+job['name'],'--output',str(dest)]
                with (dest/'build.log').open('w') as log:
                    subprocess.run(command,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[index],
                        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
            print('compiled',job['name'],flush=True)
    if args.stage=='build':
        with ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(lane,range(len(gpus))))
        return
    for condition in ['unmasked','object_context']:
        candidate=json.loads(json.dumps(recipe))
        for job in candidate['jobs']:
            if job['benchmark']!='uqis':continue
            source=root/'context_maps'/job['name'];spec=json.loads((source/'map.json').read_text())
            dest=root/condition/'maps'/job['name'];dest.mkdir(parents=True,exist_ok=True)
            for name in spec['map_files']:
                if not (dest/name).exists():os.link(source/name,dest/name)
            spec['text_context_identity']=condition
            (dest/'map.json').write_text(json.dumps(spec,indent=2)+'\n');job['scene']=str(dest/'map.json')
        candidate.update(default=False,name='Native identity with regional context: '+condition,
            comparison='Same source frames, masks, geometry and scope decoder; extra offline native descriptor channel changes text identity only',
            validation=dict(status='fixed before scoring',independent_test=False,geometry_unchanged=True,
                source_frames_unchanged=True,regions_unchanged=True,query_source_rgb=False,
                modified_modalities=['text'],public_tasks_unchanged=True))
        path=root/(condition+'_recipe.json');path.write_text(json.dumps(candidate,indent=2)+'\n')
        with (root/(condition+'_run.log')).open('w') as log:
            subprocess.run([PY,'-u','-m','radio_gs.pipeline.cohort','--recipe',str(path),
                '--output',str(root/(condition+'_system')),'--gpus',args.gpus,'--benchmarks','uqis'],
                env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print('finished',condition,flush=True)


if __name__=='__main__':main()
