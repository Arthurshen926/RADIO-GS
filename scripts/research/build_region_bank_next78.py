"""Store unmerged native descriptors and per-view sparse Gaussian memberships."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.utils.immutable_artifacts import sha256_file

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
 source=Path('output/source_coverage_next72')/a.scene;out=Path('output/region_bank_next78')/a.scene;out.mkdir(parents=True,exist_ok=True)
 data=load(Path('output/unified_response_next53/direct')/('lerf_'+a.scene)/'field.pt');gr=json.loads(Path(data['geometry_report']).read_text());gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
 path=Path('output/joint_carrier_next66/adaptive_allrgb')/a.scene/'geometry.pt';binding=dict(path=str(path),sha256=sha256_file(path));geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda');h,w=192,256
 frames=json.loads((source/'cohorts.json').read_text())['frames']['all'];entries=[]
 for frame in frames:
  row=load(source/'native_regions'/f'frame_{frame:05d}.pt');sam=load(row['sam_path']);native=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])[row['native_proposal_indices'].numpy()]
  masks=F.interpolate(torch.from_numpy(native.copy()).float()[:,None],size=(h,w),mode='nearest-exact')[:,0].flatten(1).cuda()
  cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
  renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
  hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();weight=hit['weights'].float().cuda();keep=weight>0
  ids,inverse=torch.unique(gid[keep],return_inverse=True);adj=torch.sparse_coo_tensor(torch.stack([inverse,pid[keep]]),weight[keep],(len(ids),h*w)).coalesce()
  mass=torch.sparse.mm(adj,torch.ones(h*w,1,device='cuda'))[:,0];membership=torch.sparse.mm(adj,masks.T.contiguous())/mass[:,None].clamp_min(1e-8)
  rr,cc=torch.where(membership>0);file=out/f'{frame}.pt'
  # Sparse indices are lossless int32; all positive memberships are retained.
  torch.save(dict(frame=frame,ids=ids.cpu(),mass=mass.cpu(),rows=rr.int().cpu(),columns=cc.int().cpu(),values=membership[rr,cc].cpu(),descriptors=row['descriptors'],region_indices=row['native_proposal_indices'],source=str(source/'native_regions'/f'frame_{frame:05d}.pt')),file)
  entries.append(dict(frame=frame,path=str(file),regions=len(masks),nonzero=len(rr)));print(a.scene,frame,'stored',flush=True)
 (out/'manifest.json').write_text(json.dumps(dict(geometry=binding,gaussians=len(geo.xyz),source_frames=frames,entries=entries,fit_hw=[h,w],seconds=time.monotonic()-start,query_independent=True,no_region_merging=True),indent=2)+'\n')
if __name__=='__main__':main()
