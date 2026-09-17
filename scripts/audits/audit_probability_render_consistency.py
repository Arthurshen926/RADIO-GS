"""Check full-scene colour rendering against the sealed sparse contributions."""
import argparse,json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
torch.set_num_threads(4);torch.manual_seed(381)
auth=json.loads((a.root/'responsibility.json').read_text());meta=auth['metadata'];h=meta['feature_height'];w=meta['feature_width']
g=FrozenGaussianGeometry.load(a.geometry,expected_sha256=meta['frozen_geometry_sha256'],device=a.device)
record=auth['views'][0];path=a.root/record['relative_path']
if sha256_file(path)!=record['sha256']:raise ValueError('view hash differs')
hit=torch.load(path,map_location=a.device,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weights=hit['base_weights'].float()
known=torch.rand(len(g.xyz),device=a.device)>.3
colors=torch.cat([torch.rand(len(g.xyz),3,device=a.device)*known[:,None],known[:,None].float()],dim=1)
expected=torch.zeros(h*w,4,device=a.device).index_add_(0,pid,weights[:,None]*colors[gid])
full=torch.zeros(h*w,device=a.device).scatter_add_(0,pid,weights)
camera=meta['cameras'][str(auth['frame_indices'][0])]
image,alpha,_=rasterization(means=g.xyz,quats=g.quaternions,scales=g.scales,opacities=g.opacities[:,0],colors=colors,
    viewmats=torch.tensor(camera['world_to_camera'],device=a.device)[None],Ks=torch.tensor(camera['intrinsic'],device=a.device)[None],width=w,height=h,near_plane=meta['near_plane'],far_plane=meta['far_plane'],packed=False)
actual=image[0].reshape(-1,4)
valid=(expected[:,-1]/full.clamp_min(1e-8)>.5)&(full>1e-8)
expected_response=expected[:,:3]/expected[:,-1:].clamp_min(1e-8);actual_response=actual[:,:3]/actual[:,-1:].clamp_min(1e-8)
error=float((expected_response-actual_response).abs()[valid].max())
result={'authority_sha256':sha256_file(a.root/'responsibility.json'),'geometry_sha256':meta['frozen_geometry_sha256'],'posterior_max_error':error,'weighted_color_max_error':float((expected-actual).abs().max()),'alpha_max_error':float((alpha.flatten()-full).abs().max()),'synthetic_known_element_fraction':float(known.float().mean()),'passes':error<1e-4}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
if not result['passes']:raise ValueError('rendering paths differ')
