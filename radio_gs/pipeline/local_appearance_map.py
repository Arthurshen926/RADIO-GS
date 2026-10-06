"""Retain every local native patch on the original legal observation views."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import sparse
import torch
from torch.nn import functional as F

from radio_gs.benchmarks.uqis.sensor import SensorIndex
from radio_gs.models.native_region_appearance import NativeRegionAppearance,CHECKPOINT,REPOSITORY
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks


@torch.inference_mode()
def compile_map(scene,mapping,output):
    started=time.monotonic();source=scene.parent;spec=json.loads(scene.read_text())
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    sensor=SensorIndex(mapping['sens_path']);encoder=NativeRegionAppearance()
    features=[];weights=[];pooled=[];offsets=[0]
    width=spec['configuration']['source_rgb_width']
    for i,frame in enumerate(spec['source_frames']):
        _,_,image=sensor.read(frame)
        image=image.resize((width,round(image.height*width/image.width)),Image.Resampling.LANCZOS)
        masks=torch.load(source/'source_masks'/f'{frame:06d}.pt',map_location='cpu',weights_only=False)
        masks=unpack_masks(masks['packed_masks'],width)
        local,_=encoder.patches(image)
        area=encoder.patch_weights(masks) if len(masks) else torch.zeros(0,len(local),device='cuda')
        features.append(local.half().cpu().numpy());offsets.append(offsets[-1]+len(local))
        weights.append(sparse.csr_matrix(area.cpu().numpy()))
        pooled.append(F.normalize(area@local,dim=-1).half().cpu().numpy())
        if i%25==0:print(spec['scene_id'],'patches',i+1,'/',len(spec['source_frames']),flush=True)
    for name in spec['map_files']:os.link(source/name,output/name)
    np.save(output/'patch_features.npy',np.concatenate(features))
    np.save(output/'patch_offsets.npy',np.array(offsets))
    np.save(output/'appearance_descriptors.npy',np.concatenate(pooled))
    sparse.save_npz(output/'region_patch_weights.npz',sparse.block_diag(weights,format='csr'))
    extra=['patch_features.npy','patch_offsets.npy','appearance_descriptors.npy','region_patch_weights.npz']
    report=dict(source_frames=len(features),patches=offsets[-1],regions=sum(x.shape[0] for x in weights),
        encoder=CHECKPOINT,encoder_repository=REPOSITORY,
        shared_checkpoint_bytes=Path(CHECKPOINT).stat().st_size,
        shared_parameter_bytes=sum(p.numel()*p.element_size() for p in encoder.model.parameters()),
        source_frames_added=0,patches_discarded=0,regions_discarded=0,
        geometry_and_membership_unchanged=True,query_read=False,labels_read=False,
        seconds=time.monotonic()-started)
    spec.update(parent_map=str(scene),image_native=dict(provider='dinov2_mask_pool',encoder=CHECKPOINT,
        repository=REPOSITORY,descriptors='appearance_descriptors.npy'),image_identity_fusion='equal_rank',
        local_appearance=dict(features='patch_features.npy',offsets='patch_offsets.npy',
            region_weights='region_patch_weights.npz',build=report),map_files=spec['map_files']+extra)
    spec['map_bytes']=sum((output/name).stat().st_size for name in spec['map_files'])
    (output/'map.json').write_text(json.dumps(spec,indent=2)+'\n')
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    scene=json.loads(a.scene.read_text())['scene_id']
    mapping=next(s for s in json.loads(a.mapping.read_text())['scenes'] if s['scene_id']==scene)
    compile_map(a.scene,mapping,a.output)


if __name__=='__main__':main()
