"""Reference-only ten-triplet SAM3 query construction with target read guard."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from radio_gs.querying.interactive_2d import masked_query_crop
from radio_gs.querying.positive_reference_ensemble import aggregate_positive_reference_trials
from radio_gs.querying.sam3_reference_completion import deterministic_positive_points
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model, set_requested_cuda_device, sam3_autocast_context)
from radio_gs.utils.immutable_artifacts import sha256_file

OUT=Path('output/nvos_positive_ensemble_next32')
MANIFEST=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json')


def binary(path):
    a=np.asarray(Image.open(path))
    return np.any(a>127,axis=2) if a.ndim==3 else a>127


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',nargs='+',required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    reg=json.loads((OUT/'registration.json').read_text())
    for path,digest in reg['implementations'].items():
        assert sha256_file(path)==digest,path
    manifest=json.loads(MANIFEST.read_text()); forbidden=set()
    for scene in manifest['scenes']:
        forbidden.add(str(Path(scene['prompt']['negative_path']).resolve()))
        for frame in scene['frames']:
            if frame['frame_id'] in scene['evaluation_frame_ids']:
                for key in ('rgb_path','annotation_rgb_path','gt_mask_path'):
                    if frame.get(key):forbidden.add(str(Path(frame[key]).resolve()))
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            if str(Path(os.fsdecode(args[0])).resolve()) in forbidden:
                raise RuntimeError('Target RGB/GT and negative prompts forbidden')
    sys.addaudithook(guard)
    torch.set_num_threads(4);set_requested_cuda_device(a.device)
    checkpoint=reg['checkpoint'];assert sha256_file(checkpoint)==reg['checkpoint_sha256']
    model=_load_sam3_model(checkpoint_path=checkpoint,device=a.device,
        confidence_threshold=0,dtype='float32',resolution=1008,point_only=True,build_on_cpu=True)
    for task in a.tasks:
        assert task in reg['tasks']
        start=time.time();folder=OUT/'queries'/task;folder.mkdir(parents=True,exist_ok=False)
        scene=next(s for s in manifest['scenes'] if s['scene_id']==task)
        parent_path=Path(reg['parent_query_root'])/task/'receipt.json'
        parent=json.loads(parent_path.read_text())
        assert not parent['negative_scribble_opened']
        assert sha256_file(parent_path)==reg['parent_receipts'][task]
        positive_path=parent['official_scribbles']['positive']['path']
        assert sha256_file(positive_path)==manifest['protocol']['prompt_asset_sha256'][task]['positive']
        assert sha256_file(parent['mask'])==parent['mask_sha256']
        positive=binary(positive_path);parent_mask=binary(parent['mask'])
        width,height=parent['prompt_raster_wh'];assert positive.shape==parent_mask.shape==(height,width)
        rgb=parent['reference_rgb'];assert sha256_file(rgb['path'])==rgb['sha256']
        source=Image.open(rgb['path']).convert('RGB')
        points=deterministic_positive_points(positive,count=30).reshape(10,3,2)
        masks=[];qualities=[]
        with torch.inference_mode(),sam3_autocast_context(a.device,torch.bfloat16):
            state=model.set_image(source.resize((width,height),Image.Resampling.LANCZOS))
            for coordinates in points:
                trial,quality,_=model.model.predict_inst(state,point_coords=coordinates,
                    point_labels=np.ones(3,dtype=np.int32),multimask_output=False)
                trial=np.asarray(trial).astype(bool);quality=np.asarray(quality).reshape(-1)
                assert trial.shape==(1,height,width) and quality.shape==(1,) and np.isfinite(quality).all()
                masks.append(trial[0]);qualities.append(float(quality[0]))
        trials=np.stack(masks)
        mask,probability,record=aggregate_positive_reference_trials(trials,positive,parent_mask)
        mask_path=folder/'reference_sam_mask.png';Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
        native_mask=np.asarray(Image.fromarray(mask).resize(source.size,Image.Resampling.NEAREST))
        crop,box=masked_query_crop(source,native_mask);crop_path=folder/'image_query.png';crop.save(crop_path)
        torch.save(dict(trial_masks=torch.from_numpy(trials),probability=torch.from_numpy(probability),
            positive=torch.from_numpy(positive),points=torch.from_numpy(points),qualities=qualities),folder/'trials.pt')
        receipt=dict(parent,mask=str(mask_path),mask_sha256=sha256_file(mask_path),
            image_query_crop=dict(path=str(crop_path),sha256=sha256_file(crop_path),box_xyxy=list(box)),
            parent_receipt_sha256=sha256_file(parent_path),ensemble=record,points_xy=points.tolist(),
            labels=np.ones((10,3),dtype=int).tolist(),selection=record,
            query_available=True,query_construction='positive_reference_ensemble_next32',
            trials_sha256=sha256_file(folder/'trials.pt'),registration_sha256=sha256_file(OUT/'registration.json'),
            target_gt_opened=False,negative_scribble_opened=False,seconds=time.time()-start)
        (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(task,record,flush=True)


if __name__=='__main__':main()
