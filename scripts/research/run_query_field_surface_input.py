"""Predeclared2D input-function probe on one fixed member field, all67 inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from radio_gs.query_field.runtime import QueryFieldRuntime
from radio_gs.query_field.surface_point import SurfacePointAdapter
from scripts.research import run_query_field_joint_observation as joint
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
from scripts.research.summarize_query_field_relations import paired

ROOT=Path('output/query_field_20261009/surface_input_function_v1')
PROTOCOL=Path('docs/experiments/2026-10-09-surface-input-function-protocol.md')


def prepare():
    if (joint.ROOT/'scoring_complete.json').exists() or list(joint.ROOT.glob('*/predictions/uqis/evaluation.json')):
        raise ValueError('input-function probe must be declared before any factorial score')
    ROOT.mkdir(parents=True,exist_ok=False)
    d=joint.frozen_design();c=read(d['arms']['full_supervised']['path'])
    write(ROOT/'design.json',dict(member_arm='full_supervised',member_design=receipt(joint.ROOT/'design.json'),
        member_cohort=d['arms']['full_supervised'],protocol=receipt(PROTOCOL),
        code=[receipt(Path(__file__)),receipt(Path('radio_gs/query_field/surface_point.py'))],
        package=c['uqis_package'],jobs=c['uqis'],predeclared_before_factorial_scores=True,
        additional_3d_supervision=True,model_promotion=False,modalities=['point_2d'],
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


def design():
    d=read(ROOT/'design.json')
    for r in [d['member_design'],d['member_cohort'],d['protocol']]+d['code']:check(r)
    joint.frozen_design()
    return d


@torch.no_grad()
def scene(d,index):
    read(joint.ROOT/'predictions_complete.json')
    job=d['jobs'][index];name=job['name'];public=Path(d['package'])/'method'
    queries=[q for q in read(public/'queries/point_2d.json')['queries'] if q['scene_id']==name]
    domain=next(q for q in read(public/'scenes.json')['scene_domains'] if q['scene_id']==name)
    runtime=QueryFieldRuntime.load(job['field'],'cuda');started=time.monotonic()
    adapter=SurfacePointAdapter(runtime);registration_setup=time.monotonic()-started
    mesh=np.load(public/domain['mesh_xyz_path']);_,rows=cKDTree(adapter.xyz).query(mesh,workers=4)
    rows=torch.from_numpy(rows).cuda();records=[]
    for q in queries:
        stamp=time.monotonic();packet,detail=adapter.packet(q['positive_pixel_uv'],q['camera_intrinsics'],q['camera_to_world'])
        result=runtime.query(packet);qid=q['query_id']
        score_path=ROOT/'predictions'/f'{qid}.npy';selection_path=ROOT/'selections'/f'{qid}.npy'
        gaussian_path=ROOT/'gaussians'/f'{qid}.pt'
        for p in [score_path,selection_path,gaussian_path]:
            p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():raise FileExistsError(p)
        np.save(score_path,result.membership_scores[rows].cpu().numpy().astype(np.float32))
        np.save(selection_path,result.selected[rows].cpu().numpy())
        torch.save(dict(membership=result.membership_scores.cpu(),selected=result.selected.cpu(),
            candidates=result.candidate_indices.cpu(),candidate_weights=result.candidate_weights.cpu()),gaussian_path)
        records.append(dict(query_id=qid,scene=name,modality='point_2d',adapter=detail,seconds=time.monotonic()-stamp,
            prediction=receipt(score_path),selection=receipt(selection_path),gaussians=receipt(gaussian_path)))
    write(ROOT/'receipts'/f'{name}.json',dict(records=records,field_manifest=receipt(Path(job['field'])/'manifest.json'),
        registration_setup_seconds=registration_setup,source_access=False,evaluator_access=False,unsupported=0))
    print('surface input',name,len(records),flush=True)


def predict(d,gpus):
    def action(index,gpu):
        name=d['jobs'][index]['name']
        if (ROOT/'receipts'/f'{name}.json').exists():return
        execute(['-m','scripts.research.run_query_field_surface_input','scene','--index',str(index)],ROOT/'logs'/f'{name}.log',gpu)
    lanes(list(range(9)),gpus,action)
    records=[receipt(ROOT/'receipts'/f"{j['name']}.json") for j in d['jobs']]
    count=0
    for r in records:
        for q in read(r['path'])['records']:
            for k in ['prediction','selection','gaussians']:check(q[k])
            count+=1
    if count!=67:raise ValueError('incomplete2D point cohort')
    write(ROOT/'predictions_complete.json',dict(queries=count,records=records,design=receipt(ROOT/'design.json')))


def score(d):
    check(read(ROOT/'predictions_complete.json')['design'])
    execute(['-m','radio_gs.benchmarks.uqis.evaluate','--package',d['package'],
        '--predictions',str(ROOT/'predictions'),'--selections',str(ROOT/'selections'),
        '--modalities','point_2d','--output',str(ROOT/'evaluation.json')],ROOT/'logs/score.log')
    current=read(ROOT/'evaluation.json');old=read(joint.ROOT/'full_supervised/predictions/uqis/evaluation.json')
    a={s:v['iou'] for s,v in old['explicit_selection']['modalities']['point_2d']['per_scene'].items()}
    b={s:v['iou'] for s,v in current['explicit_selection']['modalities']['point_2d']['per_scene'].items()}
    result=dict(complete=True,scope='all67 UQIS2D inputs only;not full-modality or five-task evaluation',
        member_field='full_supervised joint observation function;unchanged',extra_supervision=True,
        before_iou=float(np.mean(list(a.values()))),after_iou=float(np.mean(list(b.values()))),
        paired=paired(a,b,'UQIS point_2d IoU'),per_scene_before=a,per_scene_after=b,
        metrics=current,model_promotion=False,hard_positive_constraint=False,
        evidence=[receipt(ROOT/p) for p in ['design.json','predictions_complete.json','evaluation.json']],
        inherited_member_evaluation=receipt(joint.ROOT/'full_supervised/predictions/uqis/evaluation.json'))
    write(ROOT/'summary.json',result);write('docs/experiments/2026-10-09-surface-input-function-results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ['before_iou','after_iou','paired']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','scene','predict','score','all'])
    p.add_argument('--index',type=int);p.add_argument('--gpus',default='0,2,3');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    else:
        d=design();gpus=list(map(int,a.gpus.split(',')))
        if a.stage=='scene':scene(d,a.index)
        else:
            if a.stage in ['predict','all']:predict(d,gpus)
            if a.stage in ['score','all']:score(d)
