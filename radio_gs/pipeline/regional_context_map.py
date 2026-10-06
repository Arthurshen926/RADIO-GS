"""Keep object appearance and its visual context in separate native channels.

All original regions and legal mapping frames are encoded offline. Context
features can rank identities but never change mask membership or visibility.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from radio_gs.benchmarks.uqis.sensor import SensorIndex
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime


def crop_tensor(image, box):
    image=image.crop(box).resize((384,384),Image.Resampling.BICUBIC)
    return torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float()/255


@torch.inference_mode()
def compile_map(scene, mapping, masks_root, output, observation_range=None):
    start=time.monotonic();spec=json.loads(scene.read_text());source=scene.parent
    output.mkdir(parents=True,exist_ok=True);observations=output/'observations';observations.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    encoder=NativeSiglip2Runtime.load(Path(spec['native_encoder']),device=torch.device('cuda'))
    sensor=SensorIndex(mapping['sens_path']);width=spec['configuration']['source_rgb_width']
    batch=spec['configuration']['encoding_batch_size'];channels=dict(unmasked=[],context=[])
    for index,frame in enumerate(spec['source_frames']):
        if observation_range is not None and not observation_range[0]<=index<observation_range[1]:continue
        file=observations/f'{frame:06d}.npz'
        if not file.exists():
            _,_,image=sensor.read(frame)
            image=image.resize((width,round(image.height*width/image.width)),Image.Resampling.LANCZOS)
            payload=torch.load(masks_root/'source_masks'/f'{frame:06d}.pt',map_location='cpu',weights_only=False)
            masks=unpack_masks(payload['packed_masks'],width);crops=dict(unmasked=[],context=[])
            for mask in masks:
                y,x=np.nonzero(mask)
                box=(max(0,int(x.min())-3),max(0,int(y.min())-3),min(width,int(x.max())+4),min(image.height,int(y.max())+4))
                cx=(box[0]+box[2])*.5;cy=(box[1]+box[3])*.5
                rx=(box[2]-box[0])*.75;ry=(box[3]-box[1])*.75
                expanded=(max(0,int(np.floor(cx-rx))),max(0,int(np.floor(cy-ry))),
                          min(width,int(np.ceil(cx+rx))),min(image.height,int(np.ceil(cy+ry))))
                crops['unmasked'].append(crop_tensor(image,box))
                crops['context'].append(crop_tensor(image,expanded))
            values={name:torch.cat([encoder.encode(torch.stack(rows[i:i+batch])).half().cpu()
                for i in range(0,len(rows),batch)]).numpy() if rows else np.empty((0,1536),np.float16)
                for name,rows in crops.items()}
            temporary=observations/f'{frame:06d}.{os.getpid()}.npz'
            np.savez_compressed(temporary,**values);os.replace(temporary,file)
        values=np.load(file)
        for name in channels:channels[name].append(values[name])
        if index%20==0:print(spec['scene_id'],index+1,'/',len(spec['source_frames']),flush=True)
    # A parallel worker only compiles its assigned observations. The complete
    # scene builder still consumes every original frame before writing a map.
    if observation_range is not None:return
    for name in spec['map_files']:
        if not (output/name).exists():os.link(source/name,output/name)
    for name,rows in channels.items():np.save(output/(name+'_descriptors.npy'),np.concatenate(rows))
    extra=[name+'_descriptors.npy' for name in channels]
    spec['map_files']=spec['map_files']+extra
    report=dict(source_frames=len(spec['source_frames']),regions=len(np.concatenate(channels['context'])),
        context_expansion=1.5,encoder=spec['native_encoder'],all_original_regions_encoded=True,
        all_original_source_frames_encoded=True,query_read=False,labels_read=False,
        geometry_membership_native_descriptors_unchanged=True,seconds=time.monotonic()-start)
    spec.update(parent_map=str(scene),regional_context=dict(unmasked='unmasked_descriptors.npy',
        context='context_descriptors.npy',build=report),save_identity_evidence=True,
        map_bytes=sum((output/name).stat().st_size for name in spec['map_files']))
    (output/'map.json').write_text(json.dumps(spec,indent=2)+'\n')
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--masks-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--observation-range',type=int,nargs=2,metavar=('START','STOP'),
                   help='Parallel cache worker only; never emits a partial map')
    args=p.parse_args();scene=json.loads(args.scene.read_text())['scene_id']
    mapping=next(s for s in json.loads(args.mapping.read_text())['scenes'] if s['scene_id']==scene)
    compile_map(args.scene,mapping,args.masks_root,args.output,args.observation_range)


if __name__=='__main__':main()
