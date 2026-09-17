"""Query-free SigLIP crop grid for source pixels lacking region descriptors."""
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


@torch.inference_mode()
def native_grid_features(image_path, runtime, height, width):
    image=Image.open(image_path).convert('RGB');crops=[]
    for y in range(6):
        for x in range(8):
            for scale in (.25,.5):
                cx=(x+.5)*image.width/8;cy=(y+.5)*image.height/6
                cw,ch=round(scale*image.width),round(scale*image.height)
                left=min(max(0,round(cx-cw/2)),image.width-cw);top=min(max(0,round(cy-ch/2)),image.height-ch)
                crop=image.crop((left,top,left+cw,top+ch)).resize((384,384),Image.Resampling.BICUBIC)
                crops.append(pil_to_tensor(crop).float()/255)
    encoded=torch.cat([runtime.encode(torch.stack(crops[i:i+8]).to(runtime.device)).float() for i in range(0,len(crops),8)])
    grid=F.normalize(encoded.reshape(48,2,-1).mean(1),dim=-1).reshape(6,8,-1).permute(2,0,1)
    return F.interpolate(grid[None],size=(height,width),mode='nearest')[0].flatten(1).T.contiguous()
