"""Paired native mask-pooling / pretrained learned-mask descriptors.

Only declared source RGB and existing predicted scopes are read. The original
SigLIP prior, cameras and frozen geometry are untouched. No new detector call.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from radio_gs.models.pretrained_mask_adapter import PretrainedMaskAdapter

CHECKPOINT='/root/.cache/huggingface/hub/models--owl10--Mask-Adapter/snapshots/99662f6a64430bdb458f56f8285bd13e2bd1ac06/fcclip_convnext_large_maskadapter.pth'

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint',default=CHECKPOINT);a=p.parse_args()
    torch.set_num_threads(4);started=time.monotonic();scene=json.loads(Path(a.scene).read_text());out=Path(a.output)
    modes=['mask_pool','mask_adapter']
    for mode in modes:(out/mode).mkdir(parents=True,exist_ok=True)
    first=torch.load(scene['source_evidence']['views'][0]['identity'],map_location='cpu');texts=first['queries']
    teacher=PretrainedMaskAdapter(a.checkpoint);query=teacher.text(texts);records=[]
    for view in scene['source_evidence']['views']:
        data=torch.load(view['identity'],map_location='cpu')
        rgb=torch.from_numpy(np.array(Image.open(view['source_rgb']).convert('RGB'),copy=True)).permute(2,0,1).float()/255
        ids=[i for i,row in enumerate(data['instances']) if len(row['scores']) and (row['masks']>.5).any()]
        features={mode:torch.zeros(len(texts),768) for mode in modes};valid=torch.zeros(len(texts),dtype=torch.bool)
        if ids:
            masks=torch.stack([(data['instances'][i]['masks']>.5).any(0) for i in ids])
            values=teacher.encode(rgb,masks)
            for mode,value in values.items():features[mode][ids]=value.cpu()
            valid[ids]=True
        for mode,feature in features.items():
            probability=(100*feature.cuda()@query.T).softmax(-1).cpu()
            torch.save(dict(queries=texts,union_descriptors=feature.half(),valid=valid,class_probability=probability,
                frame=view['frame'],representation=mode,category_union_not_instance=True,encoder=a.checkpoint,
                ground_truth_read=False,external_pretrained_teacher=True),out/mode/f"frame_{view['frame']:05d}.pt")
        records.append(dict(frame=view['frame'],regions=len(ids)));print(scene['scene'],view['frame'],len(ids),flush=True)
    (out/'encoding.json').write_text(json.dumps(dict(scene=a.scene,source_views=len(records),
        new_backbone_image_encodings=sum(r['regions']>0 for r in records),new_region_encodings=sum(r['regions'] for r in records),
        new_sam_calls=0,new_text_encoder_calls=1,records=records,seconds=time.monotonic()-started,
        target_rgb_read=False,ground_truth_read=False,query_conditioned_asset=True,representations=modes,
        image_forwards_shared_between_outputs=True,checkpoint=a.checkpoint,short_edge=800,max_edge=1333,
        original_siglip_map_unchanged=True),indent=2)+'\n')
if __name__=='__main__':main()
