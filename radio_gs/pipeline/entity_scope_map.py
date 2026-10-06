"""Compile query-independent entity masks as an auxiliary scope prior.

Every legal v8 observation uses either the fixed generic SAM3 prompt 'object'
or the pretrained panoptic model's fixed upstream COCO inference configuration.
No target names, query crops, reference frames or annotations enter this build.
The original native semantic field and identity proposals remain untouched.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import sparse
import torch

from radio_gs.benchmarks.uqis.sensor import SensorIndex
from radio_gs.pipeline.registered_observation_map import visible_vertices
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model


@torch.inference_mode()
def compile_map(scene, mapping, output, provider='sam3'):
    started=time.monotonic();torch.set_num_threads(4)
    spec=json.loads(scene.read_text());source=scene.parent
    output.mkdir(parents=True,exist_ok=True)
    sensor=SensorIndex(mapping['sens_path']);xyz=np.load(source/'mesh_xyz.npy')
    if provider=='sam3':
        processor=_load_sam3_model(checkpoint_path='checkpoints/sam3_modelscope/sam3.pt',device='cuda',
            confidence_threshold=.5,dtype='float32',resolution=1008,build_on_cpu=True)
        teacher=dict(provider='SAM3 concept segmentation',generic_prompt='object',confidence_threshold=.5,
            checkpoint='checkpoints/sam3_modelscope/sam3.pt',dtype='float32 weights, bfloat16 autocast')
    else:
        from radio_gs.models.pretrained_panoptic_scope import PretrainedPanopticScope
        processor=PretrainedPanopticScope();teacher=processor.report
    matrices=[];offsets=[0];scores=[];records=[]
    width=spec['configuration']['source_rgb_width']
    for view,frame in enumerate(spec['source_frames']):
        pose,depth,rgb=sensor.read(frame)
        rgb=rgb.resize((width,round(rgb.height*width/rgb.width)),Image.Resampling.LANCZOS)
        if provider=='sam3':
            with torch.autocast('cuda',dtype=torch.bfloat16):
                state=processor.set_image(rgb)
                prediction=processor.set_text_prompt('object',state)
            masks=prediction['masks'].reshape(-1,rgb.height,rgb.width).cpu().numpy().astype(bool)
            scores.extend(prediction['scores'].float().cpu().tolist())
        else:masks=processor.masks(rgb)
        ids,u,v=visible_vertices(xyz,pose,depth,sensor,rgb.size,spec['configuration']['depth_tolerance_m'])
        region,local=np.nonzero(masks[:,v,u])
        matrices.append(sparse.csr_matrix((np.ones(len(local),np.float32),(ids[local],region)),
                                         shape=(len(xyz),len(masks))))
        offsets.append(offsets[-1]+len(masks))
        records.append(dict(frame=frame,regions=len(masks)))
        if view%20==0:print(spec['scene_id'],view+1,'/',len(spec['source_frames']),'entities',len(masks),flush=True)
    membership=sparse.hstack(matrices,format='csr')
    for name in spec['map_files']:os.link(source/name,output/name)
    sparse.save_npz(output/'entity_membership.npz',membership)
    np.save(output/'entity_offsets.npy',np.array(offsets))
    extra=['entity_membership.npz','entity_offsets.npy']
    if provider=='sam3':
        np.save(output/'entity_scores.npy',np.array(scores,np.float32));extra.append('entity_scores.npy')
    report=dict(source=str(scene),teacher=teacher,
        source_frames=len(records),regions=offsets[-1],supported_regions=int((membership.getnnz(0)>0).sum()),
        empty_frames=sum(r['regions']==0 for r in records),records=records,
        query_read=False,labels_read=False,source_frames_added=0,
        geometry_changed=False,native_features_changed=False,
        additional_teacher_path=True,
        seconds=time.monotonic()-started)
    spec.update(entity_scope=dict(membership='entity_membership.npz',offsets='entity_offsets.npy',
        build=report),map_files=spec['map_files']+extra)
    spec['map_bytes']=sum((output/name).stat().st_size for name in spec['map_files'])
    (output/'map.json').write_text(json.dumps(spec,indent=2)+'\n')
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--provider',choices=['sam3','panoptic'],default='sam3');args=p.parse_args()
    scene=json.loads(args.scene.read_text())['scene_id']
    mapping=next(s for s in json.loads(args.mapping.read_text())['scenes'] if s['scene_id']==scene)
    compile_map(args.scene,mapping,args.output,args.provider)


if __name__=='__main__':main()
