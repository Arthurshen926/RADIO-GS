"""Native query encoders independent of optional reconstruction teachers."""
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from radio_gs.utils.immutable_artifacts import sha256_file


def load_siglip_text(root, device):
    """Strictly restore checkpoint-defined output size (not hidden size)."""
    from transformers import SiglipTextConfig, SiglipTextModel, AutoTokenizer
    from safetensors import safe_open
    model = SiglipTextModel(SiglipTextConfig.from_pretrained(str(root), local_files_only=True))
    state = {}
    for shard in sorted(Path(root).glob('model-*-of-*.safetensors')):
        with safe_open(str(shard), framework='pt', device='cpu') as handle:
            for key in handle.keys():
                if key.startswith('text_model.'):
                    state[key] = handle.get_tensor(key)
    output_dim, input_dim = state['text_model.head.weight'].shape
    model.text_model.head = torch.nn.Linear(input_dim, output_dim)
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=torch.float16).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    return model, tokenizer


class NativeThreeQueryEncoder:
    """Lazy native encoders; image input is the user's crop, prompt is registered."""
    def __init__(self, identity, device='cuda:0'):
        self.identity, self.device = identity, torch.device(device)
        for record in identity['files']:
            if sha256_file(Path(record['path']).resolve()) != record['sha256']:
                raise ValueError('query encoder file identity differs: '+record['path'])
        self.text_model = self.tokenizer = self.vision = self.dino = None

    @torch.inference_mode()
    def encode_text(self, phrases):
        if self.text_model is None:
            self.text_model, self.tokenizer = load_siglip_text(self.identity['siglip_root'], self.device)
        inputs = self.tokenizer(phrases, padding='max_length', max_length=64,
                                truncation=True, return_tensors='pt').to(self.device)
        return F.normalize(self.text_model(**inputs).pooler_output.float(), dim=-1)

    @torch.inference_mode()
    def encode_image(self, crop):
        from torchvision.transforms.functional import pil_to_tensor
        from PIL import Image
        from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
        if self.vision is None:
            self.vision = NativeSiglip2Runtime.load(Path(self.identity['siglip_root']), device=self.device)
        rgb = pil_to_tensor(crop.convert('RGB').resize((384,384),Image.Resampling.BICUBIC)).float()[None]/255
        return self.vision.encode(rgb).float()

    @torch.inference_mode()
    def encode_prompt(self, image_path, seed_pixels, world_xyz, minimum, extent, height, width):
        from radio_gs.scripts.build_native_dinov2_exact_mpr_teacher import _load_model, _extract_feature_map
        if len(seed_pixels)!=2 or world_xyz.shape!=(2,3):
            raise ValueError('one positive and one negative registered seed required')
        if not torch.isfinite(world_xyz).all() or not torch.isfinite(extent).all() or (extent<=0).any():
            raise ValueError('invalid world registration')
        pixels=torch.as_tensor(seed_pixels,device=self.device,dtype=torch.long)
        if (pixels<0).any() or (pixels>=height*width).any() or pixels[0]==pixels[1]:
            raise ValueError('invalid signed pixels')
        if self.dino is None:
            self.dino=_load_model(SimpleNamespace(checkpoint=self.identity['dino_checkpoint'],
                torchhub_repo=self.identity['dino_repo'],model_name='dinov2_vitb14'),self.device)
        dense=_extract_feature_map(self.dino,Path(image_path),device=self.device,grid_height=46,grid_width=62,patch_size=14)
        # Training caches use fp16 storage before interpolation.
        dense=F.interpolate(dense.half().float()[None],size=(height,width),mode='bilinear',align_corners=False)[0].flatten(1)
        coordinates=((world_xyz.to(self.device)-minimum.to(self.device))/extent.to(self.device)).flatten()
        return torch.cat([dense[:,pixels[0]],dense[:,pixels[1]],coordinates])[None]
