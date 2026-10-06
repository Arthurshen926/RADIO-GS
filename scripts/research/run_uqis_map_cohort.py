"""Build a complete UQIS map-only cohort before opening evaluator labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

SAM='/root/miniconda3/envs/sam3/bin/python'
PY='/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/pipeline/uqis_map_only_v1.json')
    p.add_argument('--output',type=Path,default=Path('output/uqis_map_only_v1'))
    p.add_argument('--gpus',nargs='+',type=int,default=[2,3,4,5]);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());package=Path(cfg['benchmark_package'])
    mapping=json.loads((package/'mapping/scenes.json').read_text())['scenes']
    # Balance declared observations, not object identities or evaluation scores.
    mapping.sort(key=lambda s:-sum(int(f)%cfg['frame_stride']==0 for f in s['field_frame_ids']))
    lanes=[[] for _ in a.gpus];counts=[0]*len(lanes)
    for scene in mapping:
        lane=min(range(len(lanes)),key=counts.__getitem__);lanes[lane].append(scene['scene_id'])
        counts[lane]+=sum(int(f)%cfg['frame_stride']==0 for f in scene['field_frame_ids'])
    logs=a.output/'logs';logs.mkdir(parents=True,exist_ok=True)
    bound=a.output/'configuration.json'
    if bound.exists() and json.loads(bound.read_text())!=cfg:
        raise ValueError('Changed map settings require a different output directory')
    bound.write_text(json.dumps(cfg,indent=2)+'\n')
    started=time.monotonic()
    def lane_run(lane):
        gpu=a.gpus[lane];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
        for scene in lanes[lane]:
            root=a.output/'maps'/scene
            if not (root/'map.json').exists():
                for stage,python in [('masks',SAM),('compile',PY)]:
                    command=[python,'-u','-m','radio_gs.pipeline.registered_observation_map','--config',a.config,
                        '--output',str(a.output/'maps'),'--scenes',scene,'--stage',stage]
                    with (logs/f'{scene}.{stage}.log').open('w') as log:
                        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            with (logs/f'{scene}.query.log').open('w') as log:
                subprocess.run([PY,'-u','-m','radio_gs.querying.registered_observation_query','--map',str(root),
                    '--public',str(package/'method'),'--output',str(a.output)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(scene,'map and four-query predictions complete on GPU',gpu,flush=True)
    with ThreadPoolExecutor(len(lanes)) as pool:list(pool.map(lane_run,range(len(lanes))))
    prediction_record=dict(scenes=[s['scene_id'] for s in mapping],seconds=time.monotonic()-started,
        all_predictions_before_scoring=True,configuration=a.config,lanes=dict(zip(map(str,a.gpus),lanes)))
    (a.output/'predictions_complete.json').write_text(json.dumps(prediction_record,indent=2)+'\n')
    for readout in cfg['query_readouts']:
        with (logs/f'{readout}.evaluation.log').open('w') as log:
            subprocess.run([PY,'-m','radio_gs.benchmarks.uqis.evaluate','--package',str(package),
                '--predictions',str(a.output/readout/'predictions'),'--output',str(a.output/readout/'result.json')],
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    print('All scenes, four query modalities and both readouts scored.',flush=True)


if __name__=='__main__':main()
