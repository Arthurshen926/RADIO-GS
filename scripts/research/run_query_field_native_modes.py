"""Three fixed local/candidate/input function conditions, complete UQIS cohort."""
import argparse
from dataclasses import replace
import gc
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch

from radio_gs.query_field.adapters import NativeQueryEncoder,point_3d
from radio_gs.query_field.runtime import QueryFieldRuntime
from radio_gs.query_field.surface_point import SurfacePointAdapter
from radio_gs.query_field.storage import load_field,sha256
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes

ROOT=Path('output/query_field_20261009/native_mode_functions_v1')
PARENT=Path('output/query_field_20261009/joint_observation_transfer_v1')
ARMS=('mean_native','measure_resized','measure_native')
PROTOCOL=Path('docs/experiments/2026-10-09-native-mode-functions-protocol.md')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False)
    old=read(PARENT/'design.json');cohort=read(old['arms']['full_supervised']['path'])
    fields=[]
    for item in old['fields']:
        job=dict(item);job['parent']=str(PARENT/'fits'/job['scene']/'full_supervised/field')
        job['parent_manifest']=receipt(Path(job['parent'])/'manifest.json');fields.append(job)
    write(ROOT/'design.json',dict(fields=fields,package=cohort['uqis_package'],encoder=cohort['encoder'],arms=list(ARMS),
        protocol=receipt(PROTOCOL),parent_summary=receipt(PARENT/'summary.json'),parent_design=receipt(PARENT/'design.json'),
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(Path(__file__))],
        additional_instance_supervision=True,training_steps=0,model_promotion=False,whole_five_task_candidate=False,
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


def design():
    d=read(ROOT/'design.json')
    for r in [d['protocol'],d['parent_summary'],d['parent_design']]+d['code']:check(r)
    for j in d['fields']:
        check(j['parent_manifest']);check(j['legacy_manifest'])
    return d


def build(d,gpus):
    def action(i,gpu):
        name=d['fields'][i]['scene']
        if not (ROOT/'fits'/name/'construction.json').exists():
            execute(['-m','scripts.research.run_query_field_native_modes','compile','--index',str(i)],ROOT/'logs/build'/(name+'.log'),gpu)
        print('compiled',name,flush=True)
    lanes(list(range(9)),gpus,action)
    write(ROOT/'construction_complete.json',dict(exports=18,records=[receipt(ROOT/'fits'/j['scene']/'construction.json') for j in d['fields']]))


@torch.no_grad()
def scene(d,index):
    read(ROOT/'construction_complete.json');job=d['fields'][index];name=job['scene'];public=Path(d['package'])/'method'
    queries=[]
    for modality in ['text','image','point_2d','point_3d']:
        queries += [dict(q,modality=modality) for q in read(public/'queries'/(modality+'.json'))['queries'] if q['scene_id']==name]
    encoder=NativeQueryEncoder(d['encoder'],'cuda');packets={};input_receipts=[]
    for q in queries:
        if q['modality']=='text':
            packet=encoder.text(q['expression']);packets[q['query_id']]={'native':replace(packet,semantic=packet.semantic.cpu())}
        elif q['modality']=='image':
            path=public/q['crop_rgb_path']
            if sha256(path)!=q['crop_rgb_sha256']:raise ValueError('current query image changed')
            rgb=Image.open(path).convert('RGB');resized=encoder.image(rgb)
            local,mass=encoder._dino.patches(rgb)
            packets[q['query_id']]=dict(resized=replace(resized,semantic=resized.semantic.cpu(),local=resized.local.cpu()),
                native=replace(resized,semantic=resized.semantic.cpu(),local=local.cpu(),local_weights=mass.cpu()))
            input_receipts.append(dict(query_id=q['query_id'],image=receipt(path),native_patches=len(local),
                resized_patches=len(resized.local),native_mass=float(mass.sum())))
    del encoder
    packet=resized=local=mass=None
    gc.collect();torch.cuda.empty_cache()
    directory=ROOT/'fits'/name
    mean_manifest=read(directory/'mean/field/manifest.json');measure_manifest=read(directory/'measure/field/manifest.json')
    if mean_manifest['files']!=measure_manifest['files']:raise ValueError('control payloads differ beyond readout')
    field,manifest=load_field(directory/'measure/field','cuda');adapter=SurfacePointAdapter(field)
    domain=next(x for x in read(public/'scenes.json')['scene_domains'] if x['scene_id']==name)
    mesh=np.load(public/domain['mesh_xyz_path']);_,indices=cKDTree(field.xyz.cpu().numpy()).query(mesh,workers=4)
    indices=torch.from_numpy(indices).cuda()
    for arm in ARMS:
        out=ROOT/arm/'predictions/uqis';rows=[]
        if (out/'receipts'/(name+'.json')).exists():continue
        field.native_mode_rule='mean' if arm=='mean_native' else 'measure'
        current_manifest=mean_manifest if arm=='mean_native' else measure_manifest
        runtime=QueryFieldRuntime(field,current_manifest)
        for q in queries:
            qid=q['query_id'];modality=q['modality'];start=time.monotonic();registration=None
            if modality in ['text','image']:
                kind='resized' if arm=='measure_resized' and modality=='image' else 'native'
                packet=packets[qid][kind]
            elif modality=='point_2d':packet,registration=adapter.packet(q['positive_pixel_uv'],q['camera_intrinsics'],q['camera_to_world'])
            else:packet=point_3d(field,q['point_world_xyz'],maximum_distance=.1)
            result=runtime.query(packet)
            files={}
            for kind,suffix in [('predictions','.npy'),('selections','.npy'),('gaussians','.pt')]:
                p=out/kind/(qid+suffix);p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists():raise FileExistsError(p)
                files[kind]=p
            np.save(files['predictions'],result.membership_scores[indices].cpu().numpy())
            np.save(files['selections'],result.selected[indices].cpu().numpy())
            torch.save(dict(membership=result.membership_scores.cpu(),selected=result.selected.cpu(),
                candidates=result.candidate_indices.cpu(),candidate_weights=result.candidate_weights.cpu(),
                native_scores=None if result.semantic_scores is None else result.semantic_scores.cpu()),files['gaussians'])
            rows.append(dict(query_id=qid,modality=modality,status='ok',registration=registration,
                diagnostics=result.diagnostics,seconds=time.monotonic()-start,files={k:receipt(p) for k,p in files.items()}))
        write(out/'receipts'/(name+'.json'),dict(scene=name,arm=arm,queries=rows,unsupported=0,
            field_manifest=receipt(directory/('mean' if arm=='mean_native' else 'measure')/'field/manifest.json'),
            source_access=False,evaluator_access=False,current_inputs=input_receipts))
        print('predicted',arm,name,flush=True)


def verify(d):
    records=[]
    for arm in ARMS:
        ids=set()
        for j in d['fields']:
            p=ROOT/arm/'predictions/uqis/receipts'/(j['scene']+'.json');r=read(p)
            for q in r['queries']:
                if q['query_id'] in ids:raise ValueError('duplicate query')
                ids.add(q['query_id'])
                for f in q['files'].values():check(f)
            records.append(receipt(p))
        if len(ids)!=268:raise ValueError('incomplete cohort')
    return records


def predict(d,gpus):
    def action(i,gpu):
        name=d['fields'][i]['scene']
        if not all((ROOT/a/'predictions/uqis/receipts'/(name+'.json')).exists() for a in ARMS):
            execute(['-m','scripts.research.run_query_field_native_modes','scene','--index',str(i)],ROOT/'logs/predict'/(name+'.log'),gpu)
        print('all conditions predicted',name,flush=True)
    lanes(list(range(9)),gpus,action)
    write(ROOT/'predictions_complete.json',dict(queries=804,all_predictions_before_scoring=True,records=verify(d)))


def score(d,gpus):
    read(ROOT/'predictions_complete.json')
    def action(arm,gpu):
        out=ROOT/arm/'predictions/uqis'
        execute(['-m','radio_gs.benchmarks.uqis.evaluate','--package',d['package'],'--predictions',str(out/'predictions'),
            '--selections',str(out/'selections'),'--output',str(out/'evaluation.json')],ROOT/'logs/score'/(arm+'.log'),gpu)
        print('scored',arm,flush=True)
    lanes(list(ARMS),gpus,action)
    write(ROOT/'scoring_complete.json',dict(queries_per_arm=268,arms=list(ARMS)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','compile','build','scene','predict','score','all'])
    p.add_argument('--index',type=int);p.add_argument('--gpus',default='1,2,3');args=p.parse_args();torch.set_num_threads(4)
    if args.stage=='prepare':prepare()
    else:
        d=design();gpus=list(map(int,args.gpus.split(',')))
        if args.stage=='compile':
            from radio_gs.query_field.compile_native_modes import compile_scene
            j=d['fields'][args.index];compile_scene(j,j['parent'],ROOT/'fits'/j['scene'])
        elif args.stage=='scene':scene(d,args.index)
        else:
            if args.stage in ['build','all']:build(d,gpus)
            if args.stage in ['predict','all']:predict(d,gpus)
            if args.stage in ['score','all']:score(d,gpus)
