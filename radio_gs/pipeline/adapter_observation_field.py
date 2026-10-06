"""Replace only region descriptors; retain every compiled geometry weight.

Paired fixed teachers: original ConvNeXt CLIP mask pooling / pretrained
Mask-Adapter. Building this dictionary reads no query text or source detections.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from radio_gs.models.pretrained_mask_adapter import PretrainedMaskAdapter
from radio_gs.pipeline.category_union_adapter import CHECKPOINT
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--regions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);started=time.monotonic();scene=json.loads(Path(a.scene).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    teacher=PretrainedMaskAdapter(CHECKPOINT);rows={mode:[] for mode in ['mask_pool','mask_adapter']};records=[]
    rgb_paths={v['frame']:v['source_rgb'] for v in scene['source_evidence']['views']}
    for frame in scene['source_frames']:
        native=torch.load(Path(a.regions)/f'frame_{frame:05d}.pt',map_location='cpu')
        sam=torch.load(native['sam_path'],map_location='cpu')
        masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])[native['native_proposal_indices'].numpy()].copy()
        if len(masks):
            rgb=torch.from_numpy(np.array(Image.open(rgb_paths[frame]).convert('RGB'),copy=True)).permute(2,0,1).float()/255
            encoded=teacher.encode(rgb,torch.as_tensor(masks))
            for mode,value in encoded.items():rows[mode].append(value.half().cpu())
        records.append(dict(frame=frame,regions=len(masks)));print(scene['scene'],frame,len(masks),flush=True)
    field=torch.load(scene['native_observation_field'],map_location='cpu')
    for mode,features in rows.items():
        dest=out/mode;dest.mkdir(parents=True,exist_ok=True)
        feature=torch.cat(features)
        if len(feature)!=field['shape'][1]:raise ValueError('Region order/count differs from compiled transfer')
        payload={**field,'descriptors':feature,'native_space':'open_clip_convnext_large_d_320',
            'teacher_input_unchanged':False,'region_encoder':mode,'encoder_checkpoint':CHECKPOINT,
            'query_independent':True,'source_rgb_required_at_query':False}
        torch.save(payload,dest/'observation_field.pt')
        spec={**scene,'native_observation_field':str(dest/'observation_field.pt'),
            'category_query_encoder':dict(provider='mask_adapter_clip',checkpoint=CHECKPOINT)}
        spec['map_contract']={**scene['map_contract'],'native_dimension':768,'region_encoder':mode,'native_space':'open_clip_convnext_large_d_320',
            'external_pretrained_teacher':True,'region_to_element_transfer_unchanged':True}
        (dest/'scene.json').write_text(json.dumps(spec,indent=2)+'\n')
    (out/'build.json').write_text(json.dumps(dict(scene=a.scene,regions=sum(r['regions'] for r in records),source_views=len(records),
        records=records,new_backbone_image_encodings=sum(r['regions']>0 for r in records),new_sam_calls=0,
        query_text_read=False,query_conditioned_detections_read=False,ground_truth_read=False,
        image_forwards_shared_between_outputs=True,geometry_weights_unchanged=True,
        prior_field=scene['native_observation_field'],seconds=time.monotonic()-started,checkpoint=CHECKPOINT),indent=2)+'\n')
if __name__=='__main__':main()
