"""Source-only RADIO language-MLP patch extraction with overlapping windows."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.scripts.extract_radio_features import _load_radio_model
from radio_gs.models.siglip_projection import SigLIP2SummaryHead,OFFICIAL_C_RADIO_V4_H_HALF_SHA256

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def positions(length):return list(range(0,length-512,256))+[length-512]

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();torch.set_num_threads(4)
    data=load(Path('output/odds_kernel_extent_next40')/a.key/'field.pt');root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());src=json.loads((root/'source_frames.json').read_text());frames={r['frame_idx']:r for r in src['frames']}
    ckpt='/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar';model,conditioner=_load_radio_model('/root/.cache/torch/hub/NVlabs_RADIO_main',ckpt,None,torch.device('cuda'),expected_checkpoint_sha256=OFFICIAL_C_RADIO_V4_H_HALF_SHA256);head=SigLIP2SummaryHead.from_radio_checkpoint(ckpt).cuda().half().eval()
    out=Path('output/sliding_language_next46')/a.key;out.mkdir(parents=True,exist_ok=True)
    for frame in info['train_frames']:
        image=Image.open(Path(src['image_dir'])/frames[frame]['source_file']).convert('RGB');width=1024;height=max(512,round(image.height/image.width*width/16)*16)
        rgb=torch.from_numpy(np.asarray(image.resize((width,height),Image.Resampling.BICUBIC)).copy()).cuda().permute(2,0,1).float()[None]/255
        gh,gw=height//16,width//16;ref=load(Path('output/dense_language_writing_next46')/a.key/f'dense_{frame}.pt');rh,rw=ref['shape'];reference=F.normalize(F.interpolate(ref['features'].float().cuda().transpose(1,2).reshape(1,1536,rh,rw),size=(gh,gw),mode='bilinear',align_corners=False),dim=1)
        total=torch.zeros_like(reference);mass=torch.zeros(1,1,gh,gw,device='cuda');yy,xx=torch.meshgrid(torch.linspace(-1,1,32,device='cuda'),torch.linspace(-1,1,32,device='cuda'),indexing='ij');window=torch.exp(-2*(xx.square()+yy.square()))[None,None]
        for y in positions(height):
            for x in positions(width):
                with torch.autocast('cuda',dtype=torch.float16):
                    _,tokens=model(conditioner(rgb[:,:,y:y+512,x:x+512]),feature_fmt='NCHW')
                    features=head(tokens.flatten(2).transpose(1,2))
                local=F.normalize(features.float().transpose(1,2).reshape(1,1536,32,32),dim=1);y0,x0=y//16,x//16;reference_crop=reference[:,:,y0:y0+32,x0:x0+32];agreement=(local*reference_crop).sum(1,keepdim=True).clamp_min(0);weight=window*agreement
                total[:,:,y0:y0+32,x0:x0+32]+=local*weight;mass[:,:,y0:y0+32,x0:x0+32]+=weight
        value=F.normalize(torch.where(mass>0,total/mass.clamp_min(1e-8),reference),dim=1)
        torch.save(dict(features=value.flatten(2).transpose(1,2).half().cpu(),shape=[gh,gw]),out/f'dense_{frame}.pt');print(a.key,frame,'saved',gh,gw,flush=True)
    (out/'extraction.json').write_text(json.dumps(dict(backbone='C-RADIOv4-h',head='SigLIP2SummaryHead applied to raw patch tokens; experimental',window=512,stride=256,working_width=1024,blend='Gaussian times positive cosine to full-frame patch language features',query_independent=True,source_only=True,not_exact_lesv_reproduction=True),indent=2)+'\n')

if __name__=='__main__':main()
