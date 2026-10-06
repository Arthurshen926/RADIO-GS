"""Deploy the same real-observation compiler on the fixed independent rooms."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from radio_gs.benchmarks.uqis.sensor import SensorIndex
from scripts.research.prepare_referent_grounding import compile_targets

ROOT=Path('output/real_referent_grounding_20261005')
PARENT=Path('output/referent_grounding_20261005')
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
SAM_PY='/root/miniconda3/envs/sam3/bin/python'
RECIPE=Path('configs/recipes/real_referent_grounding.json')


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads((PARENT/'design.json').read_text())
    groups={k:[s for s in old[k] if (PARENT/'language'/(s+'.npz')).exists()] for k in ['fit','calibration','validation']}
    design=dict(**groups,parent=str(PARENT/'design.json'),training_condition='independent NR3D and official3D labels, real registered RGB-D observations',
        conditions=['local_scope','scene_scope','scene_identity'],epochs=12,seed=20261005,
        optimizer='AdamW lr0.0003 weight_decay0.01',logical_batch_queries=8,microbatch_queries=1,
        native_coordinates='frozen',source_frame_stride=20,maximum_source_frames=None,
        selection='calibration actual set IoU, no benchmark feedback',additional_supervision='NR3D utterances and official3D instance labels',
        targets=dict(scope='decoded-set IoU^4',identity='requested instance fraction of seed region^4; no completeness in target'),
        independent_final_test=False,default_changed=False)
    write(ROOT/'design.json',design)
    for directory in ['language','labels']:
        if not (ROOT/directory).exists():(ROOT/directory).symlink_to((PARENT/directory).resolve(),target_is_directory=True)
    (ROOT/'maps').mkdir(exist_ok=True)
    control=PARENT/'real_observation_control'
    for scene in ['scene0271_00','scene0540_00']:
        oldmap=control/'maps'/scene;new=ROOT/'maps'/scene;new.mkdir(exist_ok=True)
        spec=json.loads((oldmap/'map.json').read_text())
        for name in spec['map_files']:
            if not (new/name).exists():os.link(oldmap/name,new/name)
        write(new/'map.json',spec)
    scenes=sum(groups.values(),[])
    def download(scene):
        previous=control/'raw'/scene/(scene+'.sens')
        folder=ROOT/'raw'/scene;folder.mkdir(parents=True,exist_ok=True)
        file=folder/(scene+'.sens')
        if not file.exists():
            if previous.exists():os.link(previous,file)
            else:
                temporary=file.with_suffix('.partial')
                urllib.request.urlretrieve(f'https://kaldir.vc.in.tum.de/scannet/v1/scans/{scene}/{scene}.sens',temporary)
                temporary.rename(file)
        sensor=SensorIndex(file);frames,_=sensor.exclude([])
        print('source',scene,'frames',len(frames),'stride20',sum(int(f)%20==0 for f in frames),flush=True)
        return dict(scene_id=scene,sens_path=str(file.resolve()),field_frame_ids=frames.tolist())
    with ThreadPoolExecutor(5) as pool:mapping=list(pool.map(download,scenes))
    for folder in ['mapping','method']:(ROOT/folder).mkdir(exist_ok=True)
    write(ROOT/'mapping/scenes.json',dict(scenes=mapping))
    write(ROOT/'method/scenes.json',dict(scene_domains=[dict(scene_id=s,
        mesh_xyz_path=str(Path('output/learned_instance_additions/maps',s,'mesh_xyz.npy').resolve())) for s in scenes]))
    config=json.loads(RECIPE.read_text())['observation_compiler']
    config.update(name='complete independent real observation training cohort',benchmark_package=str(ROOT),maximum_source_frames=None)
    write(ROOT/'config.json',config)


def build(gpus):
    design=json.loads((ROOT/'design.json').read_text())
    scenes=sum([design[k] for k in ['fit','calibration','validation']],[])
    pending=[s for s in scenes if not (ROOT/'maps'/s/'map.json').exists()]
    def lane(index):
        for scene in pending[index::len(gpus)]:
            folder=ROOT/'maps'/scene;folder.mkdir(exist_ok=True)
            source=ROOT/'raw'/scene/(scene+'.sens')
            # Downloads and independent GPU compilation are pipelined.
            while not source.exists():time.sleep(1)
            sensor=SensorIndex(source);frames,_=sensor.exclude([])
            package=folder/'build_inputs'
            for name in ['mapping','method']:(package/name).mkdir(parents=True,exist_ok=True)
            write(package/'mapping/scenes.json',dict(scenes=[dict(scene_id=scene,sens_path=str(source.resolve()),field_frame_ids=frames.tolist())]))
            write(package/'method/scenes.json',dict(scene_domains=[dict(scene_id=scene,
                mesh_xyz_path=str(Path('output/learned_instance_additions/maps',scene,'mesh_xyz.npy').resolve()))]))
            config=json.loads(RECIPE.read_text())['observation_compiler']
            config.update(name='independent real observation training',benchmark_package=str(package),maximum_source_frames=None)
            write(package/'config.json',config)
            for stage,python in [('masks',SAM_PY),('compile',PY)]:
                command=[python,'-u','-m','radio_gs.pipeline.registered_observation_map','--config',str(package/'config.json'),
                    '--output',str(ROOT/'maps'),'--scenes',scene,'--stage',stage]
                with (folder/(stage+'.log')).open('w') as log:
                    subprocess.run(command,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[index],
                        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
            print('built',scene,flush=True)
    with ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(lane,range(len(gpus))))


def targets():
    queries=json.loads((PARENT/'queries.json').read_text())['queries']
    scenes=json.loads((ROOT/'design.json').read_text());scenes=sum([scenes[k] for k in ['fit','calibration','validation']],[])
    def compile_scene(scene):
        while not (ROOT/'maps'/scene/'map.json').exists():time.sleep(1)
        compile_targets(scene,[q for q in queries if q['scene']==scene],ROOT)
    with ThreadPoolExecutor(3) as pool:list(pool.map(compile_scene,scenes))


def recipes():
    for condition in ['local_scope','scene_scope','scene_identity']:
        recipe=json.loads(Path('configs/pipeline/system_v8.json').read_text())
        recipe.update(name='Real observation referent learning: '+condition,default=False,
            declaration='configs/recipes/real_referent_grounding.json',
            comparison='fixed UQIS inputs and scope; scalar referent head trained on disjoint real RGB-D rooms',
            validation=dict(status='research candidate, no promotion',independent_test=False,
                summary=str(ROOT/condition/'system/summary.json'),comparison=str(ROOT/'comparison.json')))
        for job in recipe['jobs']:
            if job['benchmark']!='uqis':continue
            parent=Path(job['scene']).parent;folder=ROOT/condition/'maps'/job['name'];folder.mkdir(parents=True,exist_ok=True)
            for file in parent.iterdir():
                if file.name!='map.json' and not (folder/file.name).exists():os.link(file,folder/file.name)
            spec=json.loads((parent/'map.json').read_text())
            spec.update(parent_map=str(parent/'map.json'),save_identity_evidence=True,
                referent_grounding=dict(checkpoint=str((ROOT/condition/'model.pt').resolve()),
                    supervision='independent NR3D expressions and official3D instance labels on real RGB-D observations',
                    context=condition.startswith('scene'),target=condition.split('_')[-1],
                    applies_to='full-expression reference-instance text queries'))
            write(folder/'map.json',spec);job['scene']=str(folder/'map.json')
        write(ROOT/(condition+'_recipe.json'),recipe)


def fit(gpus):
    conditions=['local_scope','scene_scope','scene_identity']
    def train(index):
        condition=conditions[index];context,target=condition.split('_')
        command=[PY,'-u','scripts/research/fit_referent_grounder.py','--data-root',str(ROOT),
            '--condition',context,'--target',target,'--output-name',condition,'--microbatch','1','--epochs','12']
        with (ROOT/(condition+'_fit.log')).open('w') as log:
            subprocess.run(command,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[index],
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
        print('fit complete',condition,flush=True)
    with ThreadPoolExecutor(3) as pool:list(pool.map(train,range(3)))


def run(gpus):
    conditions=['local_scope','scene_scope','scene_identity']
    while not all((ROOT/c/'training_fit_diagnosis.json').exists() for c in conditions):time.sleep(2)
    recipes()
    groups=[gpus[-2:-1],gpus[-1:],gpus[:-2]]
    def execute(index):
        condition=conditions[index]
        command=[PY,'-u','-m','radio_gs.pipeline.cohort','--recipe',str(ROOT/(condition+'_recipe.json')),
            '--output',str(ROOT/condition/'system'),'--gpus',','.join(groups[index])]
        if condition!='scene_identity':command.extend(['--benchmarks','uqis'])
        with (ROOT/(condition+'_system.log')).open('w') as log:
            subprocess.run(command,env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print('evaluated',condition,flush=True)
    with ThreadPoolExecutor(3) as pool:list(pool.map(execute,range(3)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','build','targets','recipes','fit','run'],required=True)
    p.add_argument('--gpus',default='0,1,2,4,5');a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='build':build(a.gpus.split(','))
    elif a.stage=='targets':targets()
    elif a.stage=='recipes':recipes()
    elif a.stage=='fit':fit(a.gpus.split(','))
    else:run(a.gpus.split(','))


if __name__=='__main__':main()
