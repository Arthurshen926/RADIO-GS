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
from radio_gs.query_field.storage import load_field_versioned,sha256
from radio_gs.query_field.research_storage import (research_writer,storage_binding,verify_storage_binding,validate_research_package,method_binding,runner_storage_args,prediction_binding,check_prediction_binding)
STORE_ROOT=None
LEGACY_V1=False
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes

ROOT=Path('output/query_field_20261009/native_mode_functions_v1')
PARENT=Path('output/query_field_20261009/joint_observation_transfer_v1')
ARMS=('mean_native','measure_resized','measure_native')
PROTOCOL=Path('docs/experiments/2026-10-09-native-mode-functions-protocol.md')


def prepare():
    research_writer(store_root=STORE_ROOT,legacy_v1=LEGACY_V1)
    ROOT.mkdir(parents=True,exist_ok=False)
    old=read(PARENT/'design.json');cohort=read(old['arms']['full_supervised']['path'])
    fields=[]
    for item in old['fields']:
        job=dict(item);job['parent']=str(PARENT/'fits'/job['scene']/'full_supervised/field')
        job['parent_manifest']=receipt(Path(job['parent'])/'manifest.json');fields.append(job)
    write(ROOT/'design.json',dict(storage=storage_binding(store_root=STORE_ROOT,legacy_v1=LEGACY_V1),fields=fields,package=cohort['uqis_package'],encoder=cohort['encoder'],arms=list(ARMS),
        protocol=receipt(PROTOCOL),parent_summary=receipt(PARENT/'summary.json'),parent_design=receipt(PARENT/'design.json'),
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(Path(__file__))],
        additional_instance_supervision=True,training_steps=0,model_promotion=False,whole_five_task_candidate=False,
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


def design():
    d=read(ROOT/'design.json')
    verify_storage_binding(d,store_root=STORE_ROOT,legacy_v1=LEGACY_V1)
    for r in [d['protocol'],d['parent_summary'],d['parent_design']]+d['code']:check(r)
    for j in d['fields']:
        check(j['parent_manifest']);check(j['legacy_manifest'])
    return d



def validate_compile(job):
    directory=ROOT/'fits'/job['scene'];r=read(directory/'construction.json')
    if "storage" not in r and LEGACY_V1:return r
    verify_storage_binding(r,store_root=STORE_ROOT,legacy_v1=LEGACY_V1)
    if r['parent_binding']!=method_binding(job['parent']):raise ValueError('compiled parent changed')
    identities=[]
    for entry in r['records']:
        path=directory/entry['rule']/'field'
        manifest=validate_research_package(path,store_root=STORE_ROOT,legacy_v1=LEGACY_V1)
        if entry['export_binding']!=method_binding(path,manifest):raise ValueError('compiled readout changed')
        if manifest['spec']['native_mode_rule']!=entry['rule']:raise ValueError('wrong native mode rule')
        identities.append(entry['tensor_state_identity'])
    if len(identities)!=2 or len(set(identities))!=1:raise ValueError('native control tensors differ')
    return r


def scene_binding(d,name,arm):
    if 'storage' not in d:return None
    directory=ROOT/'fits'/name/('mean' if arm=='mean_native' else 'measure')/'field'
    public=Path(d['package'])/'method'
    files=[public/'scenes.json']+[public/'queries'/(m+'.json') for m in ['text','image','point_2d','point_3d']]
    return prediction_binding(directory,d,inputs=dict(scene=name,arm=arm),query_files=files)


def verify_scene_cache(d,name,arm):
    expected=scene_binding(d,name,arm)
    if expected is None:return
    path=ROOT/arm/'predictions/uqis/receipts'/(name+'.json')
    r=read(path);check_prediction_binding(r,expected)
    if r['arm']!=arm or r['scene']!=name:raise ValueError('cached scene/arm changed')
    check(r['field_manifest'])



def build(d,gpus):
    def action(i,gpu):
        name=d['fields'][i]['scene']
        if not (ROOT/'fits'/name/'construction.json').exists():
            execute(['-m','scripts.research.run_query_field_native_modes','compile','--index',str(i)]+runner_storage_args(ROOT,store_root=STORE_ROOT,legacy_v1=LEGACY_V1),ROOT/'logs/build'/(name+'.log'),gpu)
        validate_compile(d['fields'][i])
        print('compiled',name,flush=True)
    lanes(list(range(9)),gpus,action)
    write(ROOT/'construction_complete.json',dict(exports=18,records=[receipt(ROOT/'fits'/j['scene']/'construction.json') for j in d['fields']]))


@torch.no_grad()
def scene(d,index):
    read(ROOT/'construction_complete.json');job=d['fields'][index];name=job['scene'];public=Path(d['package'])/'method'
    validate_compile(job)
    bindings={arm:scene_binding(d,name,arm) for arm in ARMS}
    for arm in ARMS:
        if (ROOT/arm/'predictions/uqis/receipts'/(name+'.json')).exists():verify_scene_cache(d,name,arm)
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
    if mean_manifest.get('tensors',mean_manifest.get('files'))!=measure_manifest.get('tensors',measure_manifest.get('files')):raise ValueError('control payloads differ beyond readout')
    field,manifest=load_field_versioned(directory/'measure/field','cuda',store_root=STORE_ROOT);adapter=SurfacePointAdapter(field)
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
        if bindings[arm]!=scene_binding(d,name,arm):raise ValueError('prediction inputs changed')
        write(out/'receipts'/(name+'.json'),dict(prediction_binding=bindings[arm],scene=name,arm=arm,queries=rows,unsupported=0,
            field_manifest=receipt(directory/('mean' if arm=='mean_native' else 'measure')/'field/manifest.json'),
            source_access=False,evaluator_access=False,current_inputs=input_receipts))
        print('predicted',arm,name,flush=True)


def verify(d):
    records=[]
    for arm in ARMS:
        ids=set()
        for j in d['fields']:
            p=ROOT/arm/'predictions/uqis/receipts'/(j['scene']+'.json');r=read(p)
            verify_scene_cache(d,j['scene'],arm)
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
            execute(['-m','scripts.research.run_query_field_native_modes','scene','--index',str(i)]+runner_storage_args(ROOT,store_root=STORE_ROOT,legacy_v1=LEGACY_V1),ROOT/'logs/predict'/(name+'.log'),gpu)
        for arm in ARMS:verify_scene_cache(d,name,arm)
        print('all conditions predicted',name,flush=True)
    lanes(list(range(9)),gpus,action)
    write(ROOT/'predictions_complete.json',dict(queries=804,all_predictions_before_scoring=True,records=verify(d)))


def score(d,gpus):
    read(ROOT/'predictions_complete.json')
    verify(d)
    def action(arm,gpu):
        out=ROOT/arm/'predictions/uqis'
        execute(['-m','radio_gs.benchmarks.uqis.evaluate','--package',d['package'],'--predictions',str(out/'predictions'),
            '--selections',str(out/'selections'),'--output',str(out/'evaluation.json')],ROOT/'logs/score'/(arm+'.log'),gpu)
        print('scored',arm,flush=True)
    lanes(list(ARMS),gpus,action)
    write(ROOT/'scoring_complete.json',dict(queries_per_arm=268,arms=list(ARMS)))


def main(argv=None):
    global ROOT,STORE_ROOT,LEGACY_V1
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','compile','build','scene','predict','score','all'])
    p.add_argument('--index',type=int);p.add_argument('--gpus',default='1,2,3');p.add_argument('--root',help='Required fresh output root for shared-v2 runs')
    p.add_argument('--store-root');p.add_argument('--legacy-v1',action='store_true')
    args=p.parse_args(argv);torch.set_num_threads(4)
    STORE_ROOT=args.store_root;LEGACY_V1=args.legacy_v1
    research_writer(store_root=STORE_ROOT,legacy_v1=LEGACY_V1)
    if not LEGACY_V1 and args.root is None:raise ValueError('new shared runs require explicit --root')
    if args.root is not None:ROOT=Path(args.root)
    if args.stage=='prepare':prepare()
    else:
        d=design();gpus=list(map(int,args.gpus.split(',')))
        if args.stage=='compile':
            from radio_gs.query_field.compile_native_modes import compile_scene
            j=d['fields'][args.index];compile_scene(j,j['parent'],ROOT/'fits'/j['scene'],store_root=STORE_ROOT,legacy_v1=LEGACY_V1)
        elif args.stage=='scene':scene(d,args.index)
        else:
            if args.stage in ['build','all']:build(d,gpus)
            if args.stage in ['predict','all']:predict(d,gpus)
            if args.stage in ['score','all']:score(d,gpus)

if __name__=='__main__':main()
