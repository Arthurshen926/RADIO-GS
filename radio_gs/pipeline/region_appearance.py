"""Add query-independent local appearance to the same frozen observation map."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from radio_gs.benchmarks.uqis.sensor import SensorIndex
from radio_gs.models.native_region_appearance import NativeRegionAppearance, CHECKPOINT, REPOSITORY
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks


@torch.inference_mode()
def compile_map(scene, mapping, output, encoder=CHECKPOINT):
    started=time.monotonic();source=scene.parent;spec=json.loads(scene.read_text())
    output.mkdir(parents=True,exist_ok=True)
    sensor=SensorIndex(mapping['sens_path']);model=NativeRegionAppearance(encoder)
    descriptors=[];records=[]
    width=spec['configuration']['source_rgb_width']
    for index,frame in enumerate(spec['source_frames']):
        _,_,image=sensor.read(frame)
        image=image.resize((width,round(image.height*width/image.width)),Image.Resampling.LANCZOS)
        mask=torch.load(source/'source_masks'/f'{frame:06d}.pt',map_location='cpu',weights_only=False)
        masks=unpack_masks(mask['packed_masks'],width)
        features=model.encode(image,masks).half().cpu().numpy() if len(masks) else np.zeros((0,model.model.embed_dim),np.float16)
        descriptors.append(features);records.append(dict(frame=frame,regions=len(masks)))
        if index%25==0:print(spec['scene_id'],'appearance',index+1,'/',len(spec['source_frames']),flush=True)
    for name in spec['map_files']:os.link(source/name,output/name)
    np.save(output/'appearance_descriptors.npy',np.concatenate(descriptors))
    report=dict(encoder=encoder,encoder_repository=REPOSITORY,
        shared_checkpoint_bytes=Path(encoder).stat().st_size,
        shared_parameter_bytes=sum(p.numel()*p.element_size() for p in model.model.parameters()),
        regions=sum(r['regions'] for r in records),source_frames=len(records),records=records,
        descriptor_rule='mask-area weighted DINOv2 native full-image patch features',
        original_geometry_masks_transfer_and_semantics_unchanged=True,
        source_frames_added=0,query_read=False,labels_read=False,source_rgb_read_at_build=True,
        source_rgb_allowed_at_query=False,seconds=time.monotonic()-started)
    spec.update(parent_map=str(scene),image_native=dict(provider='dinov2_mask_pool',encoder=encoder,
        repository=REPOSITORY,descriptors='appearance_descriptors.npy',build=report),map_files=spec['map_files']+['appearance_descriptors.npy'])
    spec['map_bytes']=sum((output/name).stat().st_size for name in spec['map_files'])
    (output/'map.json').write_text(json.dumps(spec,indent=2)+'\n')
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(spec['scene_id'],'appearance map complete',report['seconds'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--mapping',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    scene=json.loads(a.scene.read_text())['scene_id']
    mapping=next(r for r in json.loads(a.mapping.read_text())['scenes'] if r['scene_id']==scene)
    compile_map(a.scene,mapping,a.output)


if __name__=='__main__':main()
