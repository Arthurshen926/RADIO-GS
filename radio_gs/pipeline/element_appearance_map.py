"""Compile local native appearance using the frozen map's exact legal frames."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.benchmarks.uqis.sensor import SensorIndex
from .registered_observation_map import visible_vertices


@torch.inference_mode()
def compile_map(source,output):
    started=time.monotonic();spec=json.loads((source/'map.json').read_text());output.mkdir(parents=True,exist_ok=True)
    package=Path(spec['configuration']['benchmark_package'])
    mapping=next(s for s in json.loads((package/'mapping/scenes.json').read_text())['scenes'] if s['scene_id']==spec['scene_id'])
    sensor=SensorIndex(mapping['sens_path']);xyz=np.load(source/'mesh_xyz.npy')
    offsets=np.load(source/'patch_offsets.npy');patches=np.load(source/'patch_features.npy',mmap_mode='r')
    total=torch.zeros((len(xyz),patches.shape[1]),device='cuda');normsum=torch.zeros(len(xyz),device='cuda')
    count=torch.zeros(len(xyz),device='cuda');width=spec['configuration']['source_rgb_width']
    for view,frame in enumerate(spec['source_frames']):
        pose,depth,rgb=sensor.read(frame);height=round(rgb.height*width/rgb.width)
        ids,u,v=visible_vertices(xyz,pose,depth,sensor,(width,height),spec['configuration']['depth_tolerance_m'])
        local=(v//14)*((width+13)//14)+(u//14)+offsets[view]
        values=torch.tensor(patches[local].astype(np.float32),device='cuda')
        ids=torch.as_tensor(ids,device='cuda');total[ids]+=values;normsum[ids]+=values.norm(dim=1);count[ids]+=1
    coherence=total.norm(dim=1)/normsum.clamp_min(1e-8)
    np.save(output/'element_appearance.npy',F.normalize(total,dim=1).half().cpu().numpy())
    np.save(output/'element_appearance_observation.npy',torch.stack([coherence,count/len(spec['source_frames'])],dim=1).cpu().numpy())
    report=dict(scene=spec['scene_id'],source_map=str(source),native_coordinates=True,region_pooling=False,
        frames=len(spec['source_frames']),geometry_changed=False,labels_used=False,query_used=False,
        source_frames_added=0,source_frames_discarded=0,source_access_at_query_time=False,
        bytes=sum((output/name).stat().st_size for name in ['element_appearance.npy','element_appearance_observation.npy']),
        seconds=time.monotonic()-started)
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recipe',type=Path,default=Path('configs/pipeline/system_v8.json'))
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(3)
    jobs=[j for j in json.loads(a.recipe.read_text())['jobs'] if j['benchmark']=='uqis']
    for job in jobs[a.shard::a.shards]:compile_map(Path(job['scene']).parent,a.output/job['name'])


if __name__=='__main__':main()
