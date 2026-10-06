"""Project unpooled native DINO descriptors onto the unchanged training carrier."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from radio_gs.models.native_region_appearance import NativeRegionAppearance


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('output/learned_instance_additions'))
    p.add_argument('--scenes',nargs='+')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    d=json.loads((a.root/'design.json').read_text());torch.set_num_threads(3);encoder=NativeRegionAppearance()
    scenes=a.scenes or d['fit']+d['calibration']+d['validation']
    for scene in scenes[a.shard::a.shards]:
        folder=a.root/'maps'/scene;n=len(np.load(folder/'mesh_xyz.npy'));start=time.monotonic()
        total=torch.zeros((n,768),device='cuda');normsum=torch.zeros(n,device='cuda');count=torch.zeros(n,device='cuda')
        for view in range(12):
            image=Image.open(folder/f'{view:02d}.png').convert('RGB');patches,_=encoder.patches(image)
            pr=np.load(folder/f'{view:02d}.projection.npz');ids=torch.as_tensor(pr['vertices'].astype(np.int64),device='cuda')
            local=(pr['v']//14)*((image.width+13)//14)+(pr['u']//14)
            values=patches[torch.as_tensor(local.astype(np.int64),device='cuda')]
            total[ids]+=values;normsum[ids]+=values.norm(dim=1);count[ids]+=1
        coherence=total.norm(dim=1)/normsum.clamp_min(1e-8)
        np.save(folder/'element_appearance.npy',F.normalize(total,dim=1).half().cpu().numpy())
        np.save(folder/'element_appearance_observation.npy',torch.stack([coherence,count/12],dim=1).cpu().numpy())
        (folder/'element_appearance.json').write_text(json.dumps(dict(native_coordinates=True,
            region_pooling=False,frames=12,geometry_changed=False,labels_used=False,query_used=False,
            bytes=(folder/'element_appearance.npy').stat().st_size+(folder/'element_appearance_observation.npy').stat().st_size,
            seconds=time.monotonic()-start),indent=2)+'\n')
        print('projected local appearance',scene,round(time.monotonic()-start,2),flush=True)


if __name__=='__main__':main()
