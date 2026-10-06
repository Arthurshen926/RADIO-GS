"""Query-independent region support correspondence with a third source view."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
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
 source=Path('output/source_coverage_next72')/a.scene;out=Path('output/region_correspondence_next77')/a.scene;out.mkdir(parents=True,exist_ok=True)
 data=load(Path('output/unified_response_next53/direct')/('lerf_'+a.scene)/'field.pt');gr=json.loads(Path(data['geometry_report']).read_text());gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
 path=Path('output/joint_carrier_next66/adaptive_allrgb')/a.scene/'geometry.pt';geo=FrozenGaussianGeometry.load(path,expected_sha256=sha256_file(path),device='cuda');n=len(geo.xyz);h,w=192,256
 frames=json.loads((source/'cohorts.json').read_text())['frames']['all'];views={};poses=[]
 for frame in frames:
  row=load(source/'native_regions'/f'frame_{frame:05d}.pt');sam=load(row['sam_path']);native=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])[row['native_proposal_indices'].numpy()]
  masks=F.interpolate(torch.from_numpy(native.copy()).float()[:,None],size=(h,w),mode='nearest-exact')[:,0].flatten(1).cuda()
  cam=gp.cameras[frame];poses.append(torch.linalg.inv(cam.world_to_camera).numpy());k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
  renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
  hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False)
  pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();weight=hit['weights'].float().cuda();keep=weight>0
  ids,inverse=torch.unique(gid[keep],return_inverse=True);adj=torch.sparse_coo_tensor(torch.stack([inverse,pid[keep]]),weight[keep],(len(ids),h*w)).coalesce()
  mass=torch.sparse.mm(adj,torch.ones(h*w,1,device='cuda'))[:,0];membership=torch.sparse.mm(adj,masks.T.contiguous())/mass[:,None].clamp_min(1e-8)
  # Preserve all nonzero rendering contributions; no top-k region/element cap.
  rr,cc=torch.where(membership>0);matrix=sp.csr_matrix((membership[rr,cc].cpu().numpy(),(ids[rr].cpu().numpy(),cc.cpu().numpy())),shape=(n,len(masks)))
  mass_array=np.zeros(n,dtype=np.float32);mass_array[ids.cpu().numpy()]=mass.cpu().numpy()
  views[frame]=(matrix,mass_array)
  print(a.scene,frame,'projected',matrix.nnz,flush=True)
 poses=np.stack(poses);centers=poses[:,:3,3];directions=poses[:,:3,2];distance=np.linalg.norm(centers[:,None]-centers[None],axis=-1);scale=np.median(distance[distance>0]);distance=distance/scale+(1-directions@directions.T);np.fill_diagonal(distance,np.inf)
 neighbors=np.argsort(distance,axis=1)[:,:2]
 def compare(fa,fb):
  ma,wa=views[fa];mb,wb=views[fb];common=np.sqrt(wa*wb);a_mass=np.asarray(ma.T@common).ravel();b_mass=np.asarray(mb.T@common).ravel();intersection=(ma.T@mb.multiply(common[:,None])).toarray()
  dice=2*intersection/np.maximum(a_mass[:,None]+b_mass[None,:],1e-8)
  return dice,a_mass,b_mass,intersection
 result={};records=[]
 for i,fa in enumerate(frames):
  fb,fc=[frames[j] for j in neighbors[i]];ab,am,bm,inter=compare(fa,fb);ac,ac_mass,_,_=compare(fa,fc);bc,bc_mass,_,_=compare(fb,fc)
  match=ab.argmax(1);row=np.arange(len(match));pair=ab[row,match]
  # C was absent from the A-B association decision. Both matched regions must
  # support the SAME region in C, rather than choosing unrelated confirmations.
  confirmation=np.minimum(ac,bc[match]).max(1);weight=np.sqrt(np.maximum(pair*confirmation,0));unknown=(am==0)|(ac_mass==0)|(bc_mass[match]==0);weight[unknown]=1
  result[fa]=torch.from_numpy(weight.astype(np.float32))
  records.append(dict(frame=fa,association_view=fb,confirmation_view=fc,matched_regions=match.tolist(),pair_dice=pair.tolist(),third_view_support=confirmation.tolist(),a_coverage=(inter[row,match]/np.maximum(am,1e-8)).tolist(),b_coverage=(inter[row,match]/np.maximum(bm[match],1e-8)).tolist(),weights=weight.tolist(),unobserved_count=int(unknown.sum())))
 torch.save(result,out/'weights.pt');(out/'correspondence.json').write_text(json.dumps(dict(source_frames=frames,records=records,seconds=time.monotonic()-start,geometry=str(path),fit_hw=[h,w],query_independent=True,third_view_excluded_from_pair_decision=True,not_verified_instance_identity=True),indent=2)+'\n')
if __name__=='__main__':main()
