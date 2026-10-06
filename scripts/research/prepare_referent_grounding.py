"""Independent natural referring expressions on the existing frozen carriers.

Official segment/aggregation annotations establish raw NR3D object identity;
preprocessed instance integers are not assumed to preserve that identity.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy import sparse
import torch
from torch.nn import functional as F

from radio_gs.querying.observation_hypotheses import compile_hypotheses
from scripts.research.diagnose_uqis_hypothesis_capacity import selected_scopes

ROOT=Path('output/referent_grounding_20261005')
PARENT=Path('output/learned_instance_additions')
NR3D=Path('/mnt/pool/sqy/3d_understanding/ScanNet-UQIS-10/source/nr3d.csv')
ENCODER=json.loads(Path('output/local_appearance_study/chamfer/maps/scene0030_00/map.json').read_text())['native_encoder']


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def labels(scene):
    raw=ROOT/'raw'/scene;raw.mkdir(parents=True,exist_ok=True)
    for suffix in ['.aggregation.json','_vh_clean_2.0.010000.segs.json','_vh_clean_2.ply']:
        file=raw/(scene+suffix)
        if not file.exists():urllib.request.urlretrieve('https://kaldir.vc.in.tum.de/scannet/v2/scans/'+scene+'/'+file.name,file)
    v=PlyData.read(raw/(scene+'_vh_clean_2.ply'))['vertex']
    xyz=np.column_stack([v[k] for k in ['x','y','z']])
    carrier=np.load(PARENT/'maps'/scene/'mesh_xyz.npy')
    if not np.array_equal(xyz,carrier):raise ValueError(scene+': raw vertex coordinates/order differ from training carrier')
    segments=np.array(json.loads((raw/(scene+'_vh_clean_2.0.010000.segs.json')).read_text())['segIndices'])
    groups=json.loads((raw/(scene+'.aggregation.json')).read_text())['segGroups']
    lookup=np.full(int(segments.max())+1,-1,np.int32)
    for group in groups:lookup[group['segments']]=int(group['objectId'])+1
    official=lookup[segments]
    out=ROOT/'labels';out.mkdir(exist_ok=True);np.save(out/(scene+'.npy'),official)
    old=np.asarray(PlyData.read(Path('/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet/scans')/(scene+'.ply'))['vertex']['label'],np.int32)
    result=dict(scene=scene,vertices=len(xyz),coordinate_order_exact=True,
        old_ids_equal_official_fraction=float(np.mean(old==official)),objects=len(groups),
        target_id_rule='NR3D raw objectId + 1, direct official segment membership')
    print('labels',scene,result['old_ids_equal_official_fraction'],flush=True)
    return result


@torch.inference_mode()
def features(scenes):
    from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
    from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
    teacher=NativeSiglip2Runtime.load(Path(ENCODER),device=torch.device('cuda'))
    for scene in scenes:
        source=PARENT/'maps'/scene;out=ROOT/'maps'/scene;out.mkdir(parents=True,exist_ok=True)
        chunks=[]
        for frame in range(12):
            file=out/f'{frame:02d}.native.npy'
            if not file.exists():
                image=np.array(Image.open(source/f'{frame:02d}.png').convert('RGB'))
                payload=torch.load(source/f'{frame:02d}.masks.pt',map_location='cpu',weights_only=False)
                masks=unpack_masks(payload['packed_masks'],640);crops=[]
                for mask in masks:
                    y,x=np.nonzero(mask);x0,x1=max(0,int(x.min())-3),min(640,int(x.max())+4)
                    y0,y1=max(0,int(y.min())-3),min(480,int(y.max())+4)
                    crop=image[y0:y1,x0:x1].copy();crop[~mask[y0:y1,x0:x1]]=127
                    crop=Image.fromarray(crop).resize((384,384),Image.Resampling.BICUBIC)
                    crops.append(torch.from_numpy(np.array(crop,copy=True)).permute(2,0,1).float()/255)
                encoded=torch.cat([teacher.encode(torch.stack(crops[i:i+4])).half().cpu() for i in range(0,len(crops),4)]).numpy() if crops else np.zeros((0,1536),np.float16)
                np.save(file,encoded)
            chunks.append(np.load(file))
        np.save(out/'descriptors.npy',np.concatenate(chunks))
        for name in ['membership.npz','visibility.npz','region_offsets.npy','mesh_xyz.npy']:
            if not (out/name).exists():os.link(source/name,out/name)
        write(out/'build.json',dict(scene=scene,regions=sum(len(c) for c in chunks),views=12,
            native_encoder=ENCODER,labels_read=False,queries_read=False,source='original rendered RGB/masks; no proposal cap'))
        print('native',scene,sum(len(c) for c in chunks),flush=True)


def compile_targets(scene,queries,root=ROOT):
    folder=root/'maps'/scene;labels=np.load(root/'labels'/(scene+'.npy'))
    membership=sparse.load_npz(folder/'membership.npz');visibility=sparse.load_npz(folder/'visibility.npz')
    frames=np.repeat(np.arange(visibility.shape[1]),np.diff(np.load(folder/'region_offsets.npy')))
    hypothesis=compile_hypotheses(membership,visibility,frames)
    selections=selected_scopes(membership,visibility,frames,hypothesis)
    sparse.save_npz(folder/'scope_selections.npz',selections)
    targets=sorted({q['raw_object_id']+1 for q in queries});truth=np.stack([labels==t for t in targets],axis=1).astype(np.float32)
    count=truth.sum(0)
    if np.any(count==0):raise ValueError(scene+': referring object absent in official annotations')
    intersection=np.asarray(selections.T@truth);area=np.asarray(selections.sum(0)).ravel()
    iou=intersection/np.maximum(area[:,None]+count[None,:]-intersection,1)
    temporary=folder/'targets.building.npz'
    np.savez_compressed(temporary,instance_ids=np.array(targets),scope_iou=iou,
        region_intersection=np.asarray(membership.T@truth),target_size=count)
    temporary.replace(folder/'targets.npz')
    print('targets',scene,len(queries),'best scope',float(iou.max(0).mean()),flush=True)


@torch.inference_mode()
def encode_text(queries):
    from radio_gs.models.native_query_encoders import load_siglip_text
    teacher,tokenizer=load_siglip_text(ENCODER,'cuda')
    out=ROOT/'language';out.mkdir(exist_ok=True)
    scenes=list(dict.fromkeys(q['scene'] for q in queries))
    for scene in scenes:
        rows=[q for q in queries if q['scene']==scene];pooled=[];tokens=[];masks=[]
        for start in range(0,len(rows),16):
            batch=tokenizer([q['expression'] for q in rows[start:start+16]],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda')
            result=teacher(**batch)
            pooled.append(F.normalize(result.pooler_output.float(),dim=-1).half().cpu().numpy())
            tokens.append(result.last_hidden_state.half().cpu().numpy())
            masks.append((batch['input_ids']!=tokenizer.pad_token_id).cpu().numpy())
        np.savez_compressed(out/(scene+'.npz'),pooled=np.concatenate(pooled),tokens=np.concatenate(tokens),valid=np.concatenate(masks))
        write(out/(scene+'.json'),rows);print('language',scene,len(rows),flush=True)


def recipes():
    for condition in ['local','scene']:
        recipe=json.loads(Path('configs/pipeline/system_v8.json').read_text())
        recipe.update(name='RADIO-GS natural referent identity: '+condition,
            role='fixed native map and physical scope; separately declared NR3D/3D supervision',
            declaration=str(ROOT/'design.json'),default=False,
            comparison='text instance identity only; fixed field, source observations, geometry and scope; additional disjoint supervision',
            validation=dict(status='research candidate; results reported separately, no promotion',
                selection='checkpoint selected on disjoint calibration rooms only',
                independent_test=False,compact_student=False,summary=str(ROOT/condition/'system/summary.json'),
                comparison=str(ROOT/'comparison.json')))
        for job in recipe['jobs']:
            if job['benchmark']!='uqis':continue
            parent=Path(job['scene']).parent;out=ROOT/condition/'maps'/job['name']
            out.mkdir(parents=True,exist_ok=True)
            for file in parent.iterdir():
                if file.name!='map.json' and not (out/file.name).exists():os.link(file,out/file.name)
            spec=json.loads((parent/'map.json').read_text())
            spec.update(parent_map=str(parent/'map.json'),save_identity_evidence=True,
                referent_grounding=dict(checkpoint=str((ROOT/condition/'model.pt').resolve()),
                    supervision='independent NR3D natural referring expressions plus official instance labels',
                    context=condition=='scene',applies_to='full-expression reference-instance text queries'))
            write(out/'map.json',spec);job['scene']=str(out/'map.json')
        write(ROOT/(condition+'_recipe.json'),recipe)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['labels','features','targets','language','recipes'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(3);design=json.loads((ROOT/'design.json').read_text())
    scenes=design['fit']+design['calibration']+design['validation']
    split={s:k for k in ['fit','calibration','validation'] for s in design[k]}
    queries=[]
    for i,row in enumerate(csv.DictReader(NR3D.open())):
        scene=row['scan_id']
        if scene not in split:continue
        queries.append(dict(record=i,scene=scene,split=split[scene],expression=row['utterance'],
            raw_object_id=int(row['target_id']),listener_correct=row['correct_guess']=='True',category=row['instance_type']))
    if a.stage=='labels':
        with ThreadPoolExecutor(5) as pool:report=list(pool.map(labels,scenes))
        write(ROOT/'identity_alignment.json',dict(scenes=report,source='official ScanNet raw mesh/segments/aggregations',
            official_mesh_matches_each_training_vertex=True,benchmark_scenes_used=False))
        write(ROOT/'queries.json',dict(queries=queries,fit_policy='listener_correct only; all calibration/development rows scored',
            added_supervision='NR3D natural referring expressions plus independent3D instance ground truth'))
    elif a.stage=='features':features(scenes[a.shard::a.shards])
    elif a.stage=='targets':
        active=[s for s in scenes if any(q['scene']==s for q in queries)]
        with ThreadPoolExecutor(3) as pool:list(pool.map(lambda s:compile_targets(s,[q for q in queries if q['scene']==s]),active))
    elif a.stage=='language':encode_text(queries)
    else:recipes()


if __name__=='__main__':main()
