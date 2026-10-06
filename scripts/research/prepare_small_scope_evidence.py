"""Encode source SAM details discarded by the earlier coarse area cutoff."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor
from radio_gs.training.source_sam_evidence import read_manifest,load_source_sam
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.training.small_region_inventory import original_native_gray_crop,discarded_small_region_ids
from radio_gs.utils.mask_resampling import resize_boolean_masks_nearest_exact
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--base',default='output/scope_hierarchy_full_next33');a=p.parse_args();torch.set_num_threads(4)
    out=Path(a.base)/a.key;out.mkdir(parents=True,exist_ok=True)
    old=Path('output/small_region_restoration_next32')/a.key/'small_teacher.pt'
    if old.exists():shutil.copyfile(old,out/'small_teacher.pt');return
    row=next(r for r in json.loads(Path('output/native_residual_basis_next30_v2/registration.json').read_text())['cohort'] if r['key']==a.key)
    root=Path(row['root']);info=json.loads((root/'teachers.json').read_text());source=json.loads((root/'source_frames.json').read_text());authority=json.loads((root/'responsibility.json').read_text())
    manifests=[root.parent.parent/d/'source_sam/manifest.json' for d in ('base','base_full_next28')]
    manifest=next(p for p in manifests if p.exists() and sha256_file(p)==info['sam_source_manifest_sha256'])
    document,records=read_manifest(manifest);frames={r['frame_idx']:r for r in source['frames']}
    size=(authority['metadata']['feature_height'],authority['metadata']['feature_width'])
    masks=[];views=[];crops=[]
    for vi,frame in enumerate(info['frames']):
        rec=frames[frame];path=Path(source['image_dir'])/rec['source_file'];sam,_=load_source_sam(document,records,path,rec['source_sha256'])
        native=unpack_masks(sam['packed_masks'],sam['mask_shape'][1]);resized=resize_boolean_masks_nearest_exact(torch.from_numpy(native.copy()),size)
        rgb=np.asarray(Image.open(path).convert('RGB'))
        for index in discarded_small_region_ids(resized).tolist():
            crops.append(pil_to_tensor(original_native_gray_crop(rgb,native[index])).float()/255)
            masks.append(resized[index]);views.append(vi)
    teacher=torch.load(root/info['teacher_files']['siglip']['path'],map_location='cpu',weights_only=False)
    vision=NativeSiglip2Runtime.load(Path(teacher['encoder']['path']),device=torch.device('cuda'))
    descriptors=[]
    with torch.inference_mode():
        for start in range(0,len(crops),8):descriptors.append(vision.encode(torch.stack(crops[start:start+8])).cpu().half())
    torch.save(dict(masks=torch.stack(masks),proposal_views=torch.tensor(views),descriptors=torch.cat(descriptors),source_only=True),out/'small_teacher.pt')
    print(a.key,'restored',len(masks),'small source regions',flush=True)


if __name__=='__main__':main()
