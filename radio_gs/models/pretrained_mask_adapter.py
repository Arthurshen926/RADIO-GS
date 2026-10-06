"""Frozen Mask-Adapter region descriptors on its original ConvNeXt CLIP.

Inference-only port of hustvl/MaskAdapter (Apache-2.0), commit
c0516d8a548d90055c3dca7f2a9b4281a4da842f. See THIRD_PARTY_NOTICES.md.
This is an external teacher, not a newly trained RADIO-GS method or a SigLIP
adapter. The paired control pools the exact same dense backbone and input mask.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
import open_clip


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).square().mean(1, keepdim=True)
        return (x - mean) / (variance + 1e-6).sqrt() * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNextBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4*dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim)*1e-6)

    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.pwconv2(self.act(self.pwconv1(self.norm(y)))) * self.gamma
        return x + y.permute(0, 3, 1, 2)


class MaskAdapterHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = nn.Conv2d(768, 768, 1)
        self.cnext1 = ConvNextBlock(768)
        self.cnext2 = ConvNextBlock(768)
        self.cnext3 = ConvNextBlock(768)
        self.norm = nn.LayerNorm(768)
        self.final = nn.Conv2d(768, 16, 1)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1), LayerNorm2d(4), nn.GELU(),
            nn.Conv2d(4, 16, 3, stride=2, padding=1), LayerNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 768, 1))

    def forward(self, dense, masks):
        h, w = dense.shape[-2:]
        mask = F.interpolate(masks[:, None].float(), (4*h, 4*w), mode='bilinear', align_corners=False)
        x = self.fuse(dense + self.mask_downscaling(mask))
        x = self.cnext3(self.cnext2(self.cnext1(x)))
        return self.final(self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))


class PretrainedMaskAdapter:
    def __init__(self, checkpoint, device='cuda'):
        self.device = torch.device(device)
        state = torch.load(checkpoint, map_location='cpu')['model']
        self.clip = open_clip.create_model('convnext_large_d_320')
        self.clip.load_state_dict({k.removeprefix('backbone.clip_model.'): v for k, v in state.items()
                                   if k.startswith('backbone.clip_model.')}, strict=True)
        self.adapter = MaskAdapterHead()
        self.adapter.load_state_dict({k.removeprefix('mask_adapter.'): v for k, v in state.items()
                                      if k.startswith('mask_adapter.')}, strict=True)
        self.clip.to(device).eval().requires_grad_(False)
        self.adapter.to(device).eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer('convnext_large_d_320')
        self.mean = torch.tensor([122.7709383, 116.7460125, 104.09373615], device=device)[:, None, None]
        self.std = torch.tensor([68.5005327, 66.6321579, 70.32316305], device=device)[:, None, None]

    @torch.inference_mode()
    def text(self, texts):
        # Exact supplied class names; no evaluation-dependent synonyms/templates.
        return F.normalize(self.clip.encode_text(self.tokenizer(texts).to(self.device)).float(), dim=-1)

    def project(self, pooled):
        trunk = self.clip.visual.trunk
        return self.clip.visual.head(trunk.head(pooled[:, :, None, None]))

    @torch.inference_mode()
    def encode(self, rgb, masks):
        """RGB float CHW [0,1], binary masks NxHxW in the source image frame."""
        h, w = rgb.shape[-2:]
        scale = min(800/min(h, w), 1333/max(h, w))
        rh, rw = int(h*scale+.5), int(w*scale+.5)
        # ImageList convention: normalize first, zero-pad to the backbone stride.
        rgb = F.interpolate(rgb[None].to(self.device), (rh, rw), mode='bilinear', align_corners=False)[0]
        image = (rgb*255-self.mean)/self.std
        ph, pw = math.ceil(rh/32)*32, math.ceil(rw/32)*32
        image = F.pad(image, (0, pw-rw, 0, ph-rh))[None]
        masks = F.interpolate(masks[:, None].float().to(self.device), (rh, rw), mode='nearest')[:, 0]
        masks = F.pad(masks, (0, pw-rw, 0, ph-rh))
        trunk = self.clip.visual.trunk
        dense = trunk.norm_pre(trunk.stages(trunk.stem(image)))
        semantic = self.clip.visual.head(trunk.head.drop(trunk.head.norm(dense).permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        controls, learned = [], []
        for chunk in masks.split(8):
            binary = F.interpolate(chunk[:, None], dense.shape[-2:], mode='bilinear', align_corners=False)[:, 0] > 0
            weights = binary.flatten(1).float()
            weights = weights/(weights.sum(-1, keepdim=True)+1e-8)
            pooled = weights @ dense[0].flatten(1).T
            controls.append(F.normalize(self.project(pooled).float(), dim=-1))
            maps = self.adapter(semantic, chunk)
            weights = F.softmax(F.logsigmoid(maps).flatten(2), dim=-1)
            pooled = weights @ dense[0].flatten(1).T
            embedding = self.project(pooled.flatten(0, 1)).reshape(len(chunk), 16, -1).mean(1)
            learned.append(F.normalize(embedding.float(), dim=-1))
        return dict(mask_pool=torch.cat(controls), mask_adapter=torch.cat(learned))
