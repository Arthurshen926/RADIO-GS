"""Local, version-pinned aligned text/image teacher for replacement controls."""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F


class OpenClipNativeRuntime:
    def __init__(self, identity, device):
        runtime = Path(identity['runtime_root']).resolve(strict=True)
        if str(runtime) not in sys.path:
            sys.path.insert(0, str(runtime))
        import open_clip
        if open_clip.__version__ != '2.32.0':
            raise ValueError('OpenCLIP runtime version differs')
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            identity['model_name'], pretrained=identity['checkpoint_path'], device=self.device)
        self.model.eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(identity['model_name'])

    @torch.inference_mode()
    def encode_images(self, crops):
        image = torch.stack([self.preprocess(crop.convert('RGB')) for crop in crops]).to(self.device)
        return F.normalize(self.model.encode_image(image).float(), dim=-1)

    @torch.inference_mode()
    def encode_text(self, phrases):
        return F.normalize(self.model.encode_text(self.tokenizer(phrases).to(self.device)).float(), dim=-1)
