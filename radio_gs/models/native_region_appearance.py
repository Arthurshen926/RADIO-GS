"""An independent native appearance space for reference-image retrieval."""
import numpy as np
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from radio_gs.scripts.build_native_dinov2_exact_mpr_teacher import _load_model

CHECKPOINT='/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth'
REPOSITORY='/root/.cache/torch/hub/facebookresearch_dinov2_main'


class NativeRegionAppearance:
    def __init__(self, encoder=CHECKPOINT, device='cuda', repository=REPOSITORY):
        self.device=torch.device(device)
        self.model=_load_model(SimpleNamespace(torchhub_repo=repository,checkpoint=encoder,
            model_name='dinov2_vitb14'),self.device).half()
        self.patch=14
        self.mean=torch.tensor([.485,.456,.406],device=self.device)[:,None,None]
        self.std=torch.tensor([.229,.224,.225],device=self.device)[:,None,None]

    @torch.inference_mode()
    def patches(self, image):
        """Retain native local descriptors before region pooling."""
        pixels=torch.from_numpy(np.array(image.convert('RGB'),copy=True)).permute(2,0,1).to(self.device).float()/255
        h,w=pixels.shape[1:];bottom=(-h)%self.patch;right=(-w)%self.patch
        normalized=F.pad((pixels-self.mean)/self.std,(0,right,0,bottom))
        features=self.model.forward_features(normalized[None].half())['x_norm_patchtokens'][0].float()
        valid=self.patch_weights(np.ones((1,h,w),np.float32))[0]
        return features,valid

    def patch_weights(self, masks):
        h,w=masks.shape[-2:];bottom=(-h)%self.patch;right=(-w)%self.patch
        support=torch.from_numpy(masks.astype(np.float32)).to(self.device)
        return F.interpolate(F.pad(support[:,None],(0,right,0,bottom)),
            size=((h+bottom)//self.patch,(w+right)//self.patch),mode='area').flatten(1)

    @torch.inference_mode()
    def encode(self, image, masks=None):
        """Original area-weighted dense patch pooling, in unchanged coordinates.

        RGB is not stretched to a region's bounding box. Padding is excluded
        from pooling; image queries supply only their own crop, no target mask.
        """
        features,valid=self.patches(image)
        h,w=image.height,image.width
        support=np.ones((1,h,w),np.float32) if masks is None else masks.astype(np.float32)
        weights=self.patch_weights(support)
        pooled=weights@features
        return F.normalize(pooled,dim=-1)
