"""Ground automatically discovered training-image phrases; preserve native masks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model,set_requested_cuda_device,sam3_autocast_context
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.utils.native_scope_identity import native_scope_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--root',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    out=root/'caption_grounded_regions.pt'
    if out.exists():raise FileExistsError(out)
    phrasepath=root/'caption_source_phrases.json';discovery=json.loads(phrasepath.read_text())
    info=json.loads((root/'teachers.json').read_text());source=json.loads((root/'source_frames.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    assert discovery['source_only'] and not discovery['benchmark_vocabulary_used']
    assert discovery['train_frames']==info['train_frames'] and discovery['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];phrases=discovery['phrases']
    set_requested_cuda_device(a.device);checkpoint=Path('checkpoints/sam3_modelscope/sam3.pt')
    model=_load_sam3_model(checkpoint_path=str(checkpoint),device=a.device,confidence_threshold=.7,dtype='float32',resolution=1008,build_on_cpu=True)
    sources={s['frame_idx']:s for s in source['frames']};records=[];masks=[];views=[];counts={p:0 for p in phrases}
    with torch.inference_mode():
        for frame in info['train_frames']:
            src=sources[frame];imagepath=Path(source['image_dir'])/src['source_file'];assert sha256_file(imagepath)==src['source_sha256']
            image=Image.open(imagepath).convert('RGB')
            with sam3_autocast_context(a.device,torch.bfloat16):state=model.set_image(image)
            unique={}
            for qi,phrase in enumerate(phrases):
                model.reset_all_prompts(state)
                with sam3_autocast_context(a.device,torch.bfloat16):result=model.set_text_prompt(phrase,state)
                for probability,confidence in zip(result['masks_logits'],result['scores']):
                    probability=probability.float().squeeze();assert probability.shape==(image.height,image.width)
                    native=probability>.5
                    small=F.interpolate(probability[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0]>.5
                    if int(small.sum())<2 or float(small.float().mean())>.9:continue
                    counts[phrase]+=1;small=small.cpu();packed_native=torch.from_numpy(np.packbits(native.cpu().numpy(),axis=-1));key=native_scope_identity(small.numpy(),packed_native.numpy(),[image.height,image.width]);score=float(confidence)
                    if key in unique:
                        entry=unique[key];entry['phrase_indices'].append(qi)
                        if score<=entry['confidence']:continue
                        entry.update(confidence=score,packed_native=packed_native)
                    else:
                        unique[key]=dict(frame=frame,rgb_sha256=src['source_sha256'],phrase_indices=[qi],confidence=score,
                            native_shape=[image.height,image.width],packed_native=packed_native,small=small)
            for entry in unique.values():
                masks.append(entry.pop('small'));views.append(info['frames'].index(frame));records.append(entry)
            print(a.scene,frame,'distinct regions',len(unique),flush=True)
    bundle=dict(schema='caption_grounded_regions.v2',source_only=True,benchmark_data_used=False,
        masks=torch.stack(masks),proposal_views=torch.tensor(views),records=records,phrases=phrases,phrase_detection_counts=counts,
        teacher_manifest_sha256=sha256_file(root/'teachers.json'),phrase_discovery_sha256=sha256_file(phrasepath),
        geometry_binding=auth['metadata']['frozen_geometry_binding'],train_frames=info['train_frames'],
        checkpoint_sha256=sha256_file(checkpoint),implementation_sha256=sha256_file(__file__),scope_identity_implementation_sha256=sha256_file('radio_gs/utils/native_scope_identity.py'),
        policy='all source-discovered phrases on all source train frames; confidence .7; same-frame exact coarse AND native-mask dedup; preserve phrase/native-scope identity; minimum 2 coarse pixels')
    torch.save(bundle,out)
    out.with_suffix('.json').write_text(json.dumps(dict(sha256=sha256_file(out),regions=len(masks),phrase_detection_counts=counts,implementation_sha256=bundle['implementation_sha256']),indent=2)+'\n')
    print(out,flush=True)


if __name__=='__main__':main()
