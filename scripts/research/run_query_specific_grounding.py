"""Fixed matched study of source-query eligibility and bidirectional grounding."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('output/query_specific_grounding_20261005')
DATA=Path('output/real_referent_grounding_20261005')
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
CONDITIONS=['all_forward','named_forward','named_bidirectional']


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    source=list(csv.DictReader(Path('/mnt/pool/sqy/3d_understanding/ScanNet-UQIS-10/source/nr3d.csv').open()))
    for directory in ['maps','labels']:
        if not (ROOT/directory).exists():(ROOT/directory).symlink_to((DATA/directory).resolve(),target_is_directory=True)
    language=ROOT/'language';language.mkdir(exist_ok=True)
    design=json.loads((DATA/'design.json').read_text());counts={}
    for split in ['fit','calibration','validation']:
        counts[split]=[]
        for scene in design[split]:
            rows=json.loads((DATA/'language'/(scene+'.json')).read_text())
            for row in rows:row['mentions_target_class']=source[row['record']]['mentions_target_class']=='True'
            write(language/(scene+'.json'),rows)
            if not (language/(scene+'.npz')).exists():os.link(DATA/'language'/(scene+'.npz'),language/(scene+'.npz'))
            counts[split].append(dict(scene=scene,all=len(rows),listener=sum(q['listener_correct'] for q in rows),
                named=sum(q['listener_correct'] and q['mentions_target_class'] for q in rows)))
    design.update(conditions=CONDITIONS,training_condition='independent real RGB-D NR3D and official3D instance supervision; matched all/named query eligibility',
        objectives=['forward scope-quality distribution','same loss with named-target training filter','same filter with bidirectional matching'],
        training_budget='1592 query presentations each epoch; filtered arms resample eligible descriptions within each room',
        validation_policy='unchanged full207 calibration and175 development queries; no benchmark filtering',include_native_checkpoint=True)
    write(ROOT/'design.json',design);write(ROOT/'eligibility.json',counts)
    print('eligibility',json.dumps(counts),flush=True)


def fit(gpus):
    def execute(index):
        condition=CONDITIONS[index]
        command=[PY,'-u','scripts/research/fit_referent_grounder.py','--data-root',str(ROOT),'--condition','local',
            '--target','scope','--output-name',condition,'--microbatch','1','--epochs','12','--include-native-checkpoint',
            '--eligibility','listener' if index==0 else 'named','--objective','bidirectional' if index==2 else 'forward']
        env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[index],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (ROOT/(condition+'_fit.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        command=[PY,'-u','scripts/research/diagnose_referent_query_dependence.py','--checkpoint',str(ROOT/condition/'model.pt'),
            '--output',str(ROOT/(condition+'_dependence.json'))]
        with (ROOT/(condition+'_dependence.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('fit and dependence complete',condition,flush=True)
    with ThreadPoolExecutor(3) as pool:list(pool.map(execute,range(3)))


def recipes(conditions=CONDITIONS):
    for condition in conditions:
        recipe=json.loads(Path('configs/pipeline/system_v8.json').read_text())
        recipe.update(name='Query-specific grounding '+condition,default=False,declaration='configs/recipes/query_specific_grounding.json',
            comparison='fixed map, geometry, observations and scope; matched natural-language supervision study',
            validation=dict(status='research condition, not promoted',summary=str(ROOT/condition/'system/summary.json'),
                comparison=str(ROOT/'comparison.json'),independent_test=False))
        for job in recipe['jobs']:
            if job['benchmark']!='uqis':continue
            parent=Path(job['scene']).parent;folder=ROOT/condition/'maps'/job['name'];folder.mkdir(parents=True,exist_ok=True)
            for file in parent.iterdir():
                if file.name!='map.json' and not (folder/file.name).exists():os.link(file,folder/file.name)
            spec=json.loads((parent/'map.json').read_text())
            spec.update(parent_map=str(parent/'map.json'),save_identity_evidence=True,
                referent_grounding=dict(checkpoint=str((ROOT/condition/'model.pt').resolve()),
                    supervision='independent NR3D and official3D labels; '+condition,context=False,
                    applies_to='full-expression reference-instance text queries'))
            write(folder/'map.json',spec);job['scene']=str(folder/'map.json')
        write(ROOT/(condition+'_recipe.json'),recipe)


def run(gpus):
    recipes();groups=[gpus[-2:-1],gpus[-1:],gpus[:-2]]
    def execute(index):
        condition=CONDITIONS[index]
        command=[PY,'-u','-m','radio_gs.pipeline.cohort','--recipe',str(ROOT/(condition+'_recipe.json')),
            '--output',str(ROOT/condition/'system'),'--gpus',','.join(groups[index])]
        if index<2:command.extend(['--benchmarks','uqis'])
        with (ROOT/(condition+'_system.log')).open('w') as log:
            subprocess.run(command,env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print('evaluated',condition,flush=True)
    with ThreadPoolExecutor(3) as pool:list(pool.map(execute,range(3)))


def hard(gpus):
    conditions=['named_hard_forward','named_hard_bidirectional']
    design=json.loads((ROOT/'design.json').read_text());design.update(conditions=CONDITIONS+conditions,bounded_rounds=2)
    write(ROOT/'design.json',design)
    def execute(index):
        condition=conditions[index]
        env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[index],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        command=[PY,'-u','scripts/research/fit_referent_grounder.py','--data-root',str(ROOT),'--condition','local',
            '--target','scope','--output-name',condition,'--microbatch','1','--epochs','12','--include-native-checkpoint',
            '--eligibility','named','--objective','forward' if index==0 else 'bidirectional','--batch-organization','category']
        with (ROOT/(condition+'_fit.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        command=[PY,'-u','scripts/research/diagnose_referent_query_dependence.py','--checkpoint',str(ROOT/condition/'model.pt'),
            '--output',str(ROOT/(condition+'_dependence.json'))]
        with (ROOT/(condition+'_dependence.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        recipes([condition])
        command=[PY,'-u','-m','radio_gs.pipeline.cohort','--recipe',str(ROOT/(condition+'_recipe.json')),
            '--output',str(ROOT/condition/'system'),'--gpus',gpus[index],'--benchmarks','uqis']
        with (ROOT/(condition+'_system.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('hard comparison complete',condition,flush=True)
    with ThreadPoolExecutor(2) as pool:list(pool.map(execute,range(2)))


def posterior(gpu):
    condition='named_posterior'
    env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    command=[PY,'-u','scripts/research/fit_referent_grounder.py','--data-root',str(ROOT),'--condition','local',
        '--target','scope','--output-name',condition,'--microbatch','1','--epochs','12','--include-native-checkpoint',
        '--eligibility','named','--objective','bidirectional_posterior']
    with (ROOT/(condition+'_fit.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    command=[PY,'-u','scripts/research/diagnose_referent_query_dependence.py','--checkpoint',str(ROOT/condition/'model.pt'),
        '--output',str(ROOT/(condition+'_dependence.json'))]
    with (ROOT/(condition+'_dependence.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    fit_result=json.loads((ROOT/condition/'fit.json').read_text())
    base=json.loads((ROOT/'named_forward/fit.json').read_text())['validation']['scene_macro_iou']
    probe=json.loads((ROOT/(condition+'_dependence.json')).read_text())['groups']['validation']['conditions']
    gap=probe['original']['iou']-probe['query_mean']['iou'];value=fit_result['validation']['scene_macro_iou']
    passed=value>=base and gap>=.005
    gate=dict(condition=condition,calibration_iou=fit_result['calibration']['scene_macro_iou'],development_iou=value,
        named_forward_development_iou=base,query_specific_gain_pp=100*gap,minimum_query_specific_gain_pp=.5,
        passed=passed,benchmark_executed=False,decision='eligible for fixed benchmark run' if passed else
            'rejected on the declared development criteria; no benchmark run or parameter revision')
    write(ROOT/'posterior_gate.json',gate)
    print('posterior development decision',gate,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','fit','run','all','hard','posterior'],required=True)
    p.add_argument('--gpus',default='0,1,2,4,5');a=p.parse_args();gpus=a.gpus.split(',')
    if a.stage in ['prepare','all']:prepare()
    if a.stage in ['fit','all']:fit(gpus[:3])
    if a.stage in ['run','all']:run(gpus)
    if a.stage=='hard':hard(gpus)
    if a.stage=='posterior':posterior(gpus[0])
