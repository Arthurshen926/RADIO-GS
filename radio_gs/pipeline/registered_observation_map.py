"""Compile native semantics and region co-membership on registered geometry.

The builder consumes only the public mesh and legal mapping observations.
It never opens query manifests, target identities or instance annotations.
Both text/image semantics and point scope readouts use this same frozen map.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy import sparse
import torch
from torch.nn import functional as F

from radio_gs.benchmarks.uqis.sensor import SensorIndex
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model
from radio_gs.scripts.build_sam3_automatic_mask_cache import automatic_masks, unpack_masks


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def visible_vertices(xyz, pose, depth, sensor, rgb_size, tolerance):
    camera = (xyz-pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    good = np.isfinite(camera).all(1) & (z>.2) & (z<8.)
    ids = np.flatnonzero(good)
    camera = camera[ids]
    kd, kc = sensor.k_depth, sensor.k_color
    uv = camera[:, :2]/camera[:, 2, None]
    du = np.rint(uv[:, 0]*kd[0, 0]+kd[0, 2]).astype(int)
    dv = np.rint(uv[:, 1]*kd[1, 1]+kd[1, 2]).astype(int)
    cu = np.rint((uv[:, 0]*kc[0, 0]+kc[0, 2])*rgb_size[0]/sensor.color_width).astype(int)
    cv = np.rint((uv[:, 1]*kc[1, 1]+kc[1, 2])*rgb_size[1]/sensor.color_height).astype(int)
    good = (du>=0)&(du<sensor.width)&(dv>=0)&(dv<sensor.height)&(cu>=0)&(cu<rgb_size[0])&(cv>=0)&(cv<rgb_size[1])
    ids, du, dv, cu, cv = [v[good] for v in (ids,du,dv,cu,cv)]
    good = (depth[dv,du]>.2)&(np.abs(depth[dv,du]-z[ids])<=tolerance)
    return ids[good], cu[good], cv[good]


def mask_arguments(config):
    return SimpleNamespace(grid_size=config['sam_grid_size'], minimum_quality=config['sam_minimum_quality'],
        minimum_area_fraction=config['sam_minimum_area_fraction'],maximum_area_fraction=config['sam_maximum_area_fraction'],
        minimum_stability=0.,stability_offset=1.,nms_iou=.85,duplicate_minimum_area_ratio=.9,
        maximum_masks=config['sam_maximum_masks'])


@torch.inference_mode()
def build_masks(mapping, config, output, processor):
    folder=output/'source_masks';folder.mkdir(parents=True,exist_ok=True)
    sensor=SensorIndex(mapping['sens_path'])
    frames=[int(f) for f in mapping['field_frame_ids'] if int(f)%config['frame_stride']==0]
    for n,frame in enumerate(frames):
        path=folder/f'{frame:06d}.pt'
        if path.exists():continue
        start=time.monotonic();_,_,rgb=sensor.read(frame)
        width=config['source_rgb_width'];rgb=rgb.resize((width,round(rgb.height*width/rgb.width)),Image.Resampling.LANCZOS)
        payload=automatic_masks(processor,rgb,mask_arguments(config))
        payload.update(frame=frame,query_independent=True,seconds=time.monotonic()-start)
        torch.save(payload,path)
        print(mapping['scene_id'],'masks',n+1,'/',len(frames),len(payload['scores']),round(payload['seconds'],2),flush=True)


@torch.inference_mode()
def build_scene(mapping, domain, public_root, config, output, encoder):
    started=time.monotonic(); output.mkdir(parents=True,exist_ok=True)
    if (output/'map.json').exists():
        return json.loads((output/'map.json').read_text())
    observations=output/'observations'; observations.mkdir(exist_ok=True)
    xyz=np.load(public_root/domain['mesh_xyz_path'])
    sensor=SensorIndex(mapping['sens_path'])
    frames=[int(f) for f in mapping['field_frame_ids'] if int(f)%config['frame_stride']==0]
    matrices=[]; descriptors=[]; visible_rows=[]; visible_cols=[]; offsets=[0]; records=[]
    for view_index, frame in enumerate(frames):
        file=observations/f'{frame:06d}.pt'; stamp=time.monotonic()
        if file.exists():
            row=torch.load(file,map_location='cpu',weights_only=False)
        else:
            pose,depth,rgb=sensor.read(frame)
            width=config['source_rgb_width'];height=round(rgb.height*width/rgb.width)
            rgb=rgb.resize((width,height),Image.Resampling.LANCZOS)
            payload=torch.load(output/'source_masks'/f'{frame:06d}.pt',map_location='cpu',weights_only=False)
            masks=unpack_masks(payload['packed_masks'],width)
            image=np.asarray(rgb); crops=[]
            for mask in masks:
                y,x=np.nonzero(mask)
                x0,x1=max(0,int(x.min())-3),min(width,int(x.max())+4)
                y0,y1=max(0,int(y.min())-3),min(height,int(y.max())+4)
                crop=image[y0:y1,x0:x1].copy();crop[~mask[y0:y1,x0:x1]]=127
                crops.append(torch.from_numpy(np.array(Image.fromarray(crop).resize((384,384),Image.Resampling.BICUBIC),copy=True)).permute(2,0,1).float()/255)
            batch=config['encoding_batch_size']
            encoded=torch.cat([encoder.encode(torch.stack(crops[i:i+batch])).half().cpu() for i in range(0,len(crops),batch)]) if crops else torch.empty(0,1536,dtype=torch.float16)
            ids,u,v=visible_vertices(xyz,pose,depth,sensor,(width,height),config['depth_tolerance_m'])
            region,local=np.nonzero(masks[:,v,u])
            row=dict(frame=frame,descriptors=encoded,element_indices=torch.from_numpy(ids[local]),
                region_indices=torch.from_numpy(region),visible_vertices=torch.from_numpy(ids),
                seconds=time.monotonic()-stamp,query_independent=True)
            torch.save(row,file)
        count=len(row['descriptors']); descriptors.append(row['descriptors'])
        matrix=sparse.coo_matrix((np.ones(len(row['element_indices']),np.float32),
            (row['element_indices'].numpy(),row['region_indices'].numpy())),shape=(len(xyz),count)).tocsr()
        matrices.append(matrix)
        visible_rows.append(row['visible_vertices'].numpy());visible_cols.append(np.full(len(visible_rows[-1]),view_index,np.int32))
        offsets.append(offsets[-1]+count)
        records.append(dict(frame=frame,regions=count,visible_vertices=len(visible_rows[-1]),seconds=row['seconds']))
        print(mapping['scene_id'],view_index+1,'/',len(frames),'regions',count,'seconds',round(row['seconds'],2),flush=True)
    membership=sparse.hstack(matrices,format='csr')
    visibility=sparse.coo_matrix((np.ones(sum(map(len,visible_rows)),np.float32),
        (np.concatenate(visible_rows),np.concatenate(visible_cols))),shape=(len(xyz),len(frames))).tocsr()
    # These weights retain view balance; Gaussian contribution mass is a
    # different adapter policy and must not silently replace this operation.
    from radio_gs import evidence_core
    if evidence_core.enabled():
        transfer=evidence_core.view_balanced_transfer(membership,np.asarray(offsets))
        mass=np.asarray(transfer.sum(1)).reshape(-1)
    else:
        mixed=[]
        for matrix in matrices:
            count=np.asarray(matrix.sum(1)).reshape(-1)
            mixed.append(sparse.diags(1/np.maximum(count,1))@matrix)
        transfer=sparse.hstack(mixed,format='csr')
        mass=np.asarray(transfer.sum(1)).reshape(-1)
        transfer=(sparse.diags(1/np.maximum(mass,1))@transfer).tocsr()
    sparse.save_npz(output/'membership.npz',membership)
    sparse.save_npz(output/'visibility.npz',visibility)
    sparse.save_npz(output/'transfer.npz',transfer)
    np.save(output/'descriptors.npy',torch.cat(descriptors).numpy())
    np.save(output/'mesh_xyz.npy',xyz)
    np.save(output/'region_offsets.npy',np.array(offsets))
    report=dict(schema='registered_native_observation_map_v1',scene_id=mapping['scene_id'],
        elements=len(xyz),regions=offsets[-1],source_frames=frames,source_count=len(frames),
        supported_elements=int((mass>0).sum()),mapping_visibility_elements=int((visibility.getnnz(axis=1)>0).sum()),
        map_files=['membership.npz','visibility.npz','transfer.npz','descriptors.npy','mesh_xyz.npy','region_offsets.npy'],
        native_encoder=config['native_encoder'],configuration=config,query_independent=True,
        labels_used=False,query_manifest_read=False,source_access_at_query_time=False,
        compact_student=False,seconds=time.monotonic()-started,records=records)
    report['map_bytes']=sum((output/p).stat().st_size for p in report['map_files'])
    write_json(output/'map.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenes',nargs='+',required=True)
    p.add_argument('--stage',choices=['masks','compile'],required=True)
    a=p.parse_args();torch.set_num_threads(4)
    cfg=json.loads(a.config.read_text());package=Path(cfg['benchmark_package'])
    mapping=json.loads((package/'mapping/scenes.json').read_text())['scenes']
    domains=json.loads((package/'method/scenes.json').read_text())['scene_domains']
    if a.stage=='masks':
        processor=_load_sam3_model(checkpoint_path=cfg['sam_checkpoint'],device='cuda',confidence_threshold=0.,
            dtype='bfloat16',resolution=cfg['sam_resolution'],point_only=True)
    else:
        from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
        encoder=NativeSiglip2Runtime.load(Path(cfg['native_encoder']),device=torch.device('cuda'))
    for scene in a.scenes:
        if a.stage=='masks':
            build_masks(next(s for s in mapping if s['scene_id']==scene),cfg,a.output/scene,processor)
            continue
        build_scene(next(s for s in mapping if s['scene_id']==scene),next(d for d in domains if d['scene_id']==scene),
            package/'method',cfg,a.output/scene,encoder)


if __name__=='__main__':main()
