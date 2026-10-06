"""Offline panoptic scope from the already cached FC-CLIP/Mask-Adapter model.

Uses the upstream COCO inference configuration and full pretrained weights.
Only the predicted masks enter the map; its category names never replace the
query's native semantic features. This is an external prior, not our training.
"""
import os
from pathlib import Path
import sys

import numpy as np
import torch

from radio_gs.pipeline.category_union_adapter import CHECKPOINT

REPOSITORY='/root/baselines/MaskAdapter'
DETECTRON_SITE='/root/miniconda3/envs/iclpose/lib/python3.9/site-packages'
CONFIG='configs/mixed-mask-training/fc-clip/fcclip/fcclip_convnext_large_eval_coco.yaml'


class PretrainedPanopticScope:
    def __init__(self):
        # Reuse the installed CPython 3.9 Detectron extension with this runtime's
        # torch; no global environment or upstream repository is modified.
        sys.path.append(DETECTRON_SITE)
        sys.path.insert(0,REPOSITORY)
        previous=Path.cwd()
        os.chdir(REPOSITORY)
        try:
            import fcclip
            from detectron2.config import get_cfg
            from detectron2.modeling import build_model
            from detectron2.projects.deeplab import add_deeplab_config
            from detectron2.data.transforms import ResizeShortestEdge
            cfg=get_cfg();add_deeplab_config(cfg)
            fcclip.add_maskformer2_config(cfg);fcclip.add_fcclip_config(cfg);fcclip.add_mask_adapter_config(cfg)
            cfg.merge_from_file(CONFIG)
            # Complete weights below include the CLIP backbone; avoid another
            # download and overwrite every tensor with the cached checkpoint.
            cfg.MODEL.FC_CLIP.CLIP_PRETRAINED_WEIGHTS=''
            cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON=False
            cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON=False
            cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON=True
            self.model=build_model(cfg).eval().requires_grad_(False)
            state=torch.load(CHECKPOINT,map_location='cpu')['model']
            # The released file also stores a training-only loss class weight.
            state={k:v for k,v in state.items() if not k.startswith('criterion.')}
            self.model.load_state_dict(state,strict=True)
            self.resize=ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST,cfg.INPUT.MIN_SIZE_TEST],cfg.INPUT.MAX_SIZE_TEST)
            self.format=cfg.INPUT.FORMAT
            self.report=dict(repository=REPOSITORY,checkpoint=CHECKPOINT,configuration=CONFIG,
                provider='FC-CLIP with Mask-Adapter, upstream COCO panoptic inference',
                vocabulary='fixed upstream COCO, not benchmark or query names',
                object_mask_threshold=float(cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD),
                overlap_threshold=float(cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD),
                model_object_queries=int(cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES),additional_mask_cap=None)
        finally:
            os.chdir(previous)

    @torch.inference_mode()
    def masks(self,image):
        rgb=np.asarray(image)
        values=rgb[:,:,::-1] if self.format=='BGR' else rgb
        values=self.resize.get_transform(values).apply_image(values)
        tensor=torch.as_tensor(np.ascontiguousarray(values.transpose(2,0,1))).float()
        result=self.model([dict(image=tensor,height=image.height,width=image.width)])[0]
        labels,segments=result['panoptic_seg']
        masks=torch.stack([labels==s['id'] for s in segments]) if segments else labels.new_empty((0,*labels.shape),dtype=torch.bool)
        return masks.cpu().numpy()
