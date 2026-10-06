"""Area-ordered native region scale slots on the unchanged Gaussian carrier."""
import argparse,json,shutil,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.querying.typed_evidence import visible_operator
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--arm',default='adaptive_allrgb');p.add_argument('--cohort',choices=['old','coverage64','all'],default='all');p.add_argument('--fit-width',type=int,default=128);p.add_argument('--output-base',default='output/expanded_semantic_next72');p.add_argument('--residual-from');p.add_argument('--region-mode',choices=['leaf','hierarchy'],required=True);a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
 parent=Path('output/unified_response_next53/direct')/a.key;data=load(parent/'field.pt');online=load(parent/'typed_readout.pt');root=Path(data['root']);gr=json.loads(Path(data['geometry_report']).read_text());gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);from radio_gs.utils.immutable_artifacts import sha256_file
 geometry_path=Path('output/joint_carrier_next66')/a.arm/a.key.removeprefix('lerf_')/'geometry.pt';binding=dict(path=str(geometry_path),sha256=sha256_file(geometry_path));geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
 info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
 ch,cw=h,w
 evidence=Path('output/source_coverage_next72')/a.key.removeprefix('lerf_');source_frames=json.loads((evidence/'cohorts.json').read_text())['frames'][a.cohort];h=round(h*a.fit_width/w);w=a.fit_width
 from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
 q=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].half().float().cuda(),dim=1)
 observed=torch.ones(len(geo.xyz),dtype=torch.bool,device='cuda');n=len(observed);d=q.shape[1];examples=[];modes=['leaf'] if a.region_mode=='leaf' else ['mixture','leaf','parent'];sums={m:torch.zeros(n,d) for m in modes};mass={m:torch.zeros(n,1,device='cuda') for m in sums}
 initial_features=F.normalize(load(a.residual_from)['features'].float(),dim=1) if a.residual_from else None
 with torch.no_grad():
  for frame in source_frames:
   row=load(evidence/'native_regions'/f'frame_{frame:05d}.pt');sam=load(row['sam_path']);native=unpack_masks(sam['packed_masks'],sam['mask_shape'][1]);mask=F.interpolate(torch.from_numpy(native[row['native_proposal_indices'].numpy()].copy()).float()[:,None],size=(h,w),mode='nearest-exact')[:,0].flatten(1).cuda();count=mask.sum(0);desc=F.normalize(row['descriptors'].float().cuda(),dim=1)
   mix=F.normalize(mask.T@desc/count[:,None].clamp_min(1),dim=1)
   # Area orders overlapping supports, not object identity. Keep native vectors
   # in separate scale slots; never average leaf and enclosure before matching.
   areas=torch.from_numpy(native[row['native_proposal_indices'].numpy()].sum(axis=(1,2))).float().cuda()
   leaf=(areas[:,None].expand_as(mask).masked_fill(mask==0,float('inf'))).argmin(0)
   coarse_index=(areas[:,None].expand_as(mask).masked_fill(mask==0,-1)).argmax(0)
   grids={'mixture':mix,'leaf':desc[leaf]*(count>0)[:,None],'parent':desc[coarse_index]*(count>0)[:,None]}
   cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
   renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
   hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights'];op,_,valid=visible_operator(hit,observed,h*w,False)
   pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();weight=hit['weights'].float().cuda()*(count[pid]>0)
   if initial_features is not None:
    total=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,hit['weights'].float().cuda());weight=weight/total[pid].clamp_min(1e-8)
   # The adjoint has zero rows for all untouched Gaussians. Reduce only
   # its nonzero rows, then scatter them into the SAME full native field.
   keep=weight>0;touched,inverse=torch.unique(gid[keep],sorted=True,return_inverse=True);touched_cpu=touched.cpu()
   adj=torch.sparse_coo_tensor(torch.stack([inverse,pid[keep]]),weight[keep],(len(touched),h*w)).coalesce()
   for mode in modes:
    grid=grids[mode]
    for channel in range(0,d,128):
     target=grid[:,channel:channel+128].contiguous()
     if initial_features is not None:target=target-torch.sparse.mm(adj.T.coalesce(),initial_features[touched_cpu,channel:channel+128].cuda())
     sums[mode][touched_cpu,channel:channel+128]+=torch.sparse.mm(adj,target).cpu()
    mass[mode].index_add_(0,touched,torch.sparse.mm(adj,torch.ones(h*w,1,device='cuda')))
   if frame not in info['train_frames']:continue
   # Keep calibration evidence and raster fixed: isolate native feature writing.
   if (h,w)!=(ch,cw):
    ck=cam.intrinsic.cuda().clone();ck[0]*=cw/cam.width;ck[1]*=ch/cam.height
    cr=SimpleNamespace(scaled_intrinsics=lambda width,height:ck,near_plane=.001*gp.scale,far_plane=100*gp.scale)
    calibration_hit=rasterize_single_view_contributions(geo,cr,cam.world_to_camera.cuda(),height=ch,width=cw,include_depths=False);calibration_hit['base_weights']=calibration_hit['weights'];op,_,valid=visible_operator(calibration_hit,observed,ch*cw,False)
   record=load(Path('output/typed_evidence_next44')/a.key/f'identity_{frame}.pt')
   target=torch.stack([F.interpolate(r['masks'].float().amax(0)[None,None],size=(ch,cw),mode='bilinear',align_corners=False)[0,0] if len(r['scores']) else torch.zeros(ch,cw) for r in record['instances']],-1).reshape(ch*cw,-1).cuda()
   active=torch.tensor([len(r['scores'])>0 for r in record['instances']],device='cuda');binary=(target.T.reshape(-1,1,ch,cw)>.5).float();pos=(-F.max_pool2d(-binary,3,1,1)>.5).flatten(1).T&valid;neg=(F.max_pool2d(binary,3,1,1)<.5).flatten(1).T&valid
   examples.append((op,target,pos&active,neg&active));print(a.key,frame,'features fused',flush=True)
 if initial_features is not None:
  for mode in sums:
   sums[mode].div_(mass[mode].cpu().clamp_min(1e-8)).mul_(.95).add_(initial_features)
  del initial_features
 # Match the same native query against each scale separately. A single
 # shared scalar calibration consumes the union score, not per-query gates.
 cosine_slots=[]
 for mode in modes:
  features=F.normalize(sums[mode],dim=1)
  cosine_slots.append(torch.cat([part.cuda()@q.T for part in features.split(8192)]).detach())
 cosine=torch.stack(cosine_slots).amax(0);supported=mass[modes[0]]>0
 for mode in [a.region_mode]:
  # Two scalar calibration parameters, trained only on cached source detections.
  log_scale=torch.nn.Parameter(torch.tensor(3.,device='cuda'));bias=torch.nn.Parameter(torch.tensor(-5.,device='cuda'));opt=torch.optim.Adam([log_scale,bias],lr=.03)
  for step in range(240):
   op,y,pos,neg=examples[step%len(examples)];value=(log_scale.exp()*cosine+bias).sigmoid()*supported;pred=torch.sparse.mm(op,value);known=pos|neg;loss=((pred-y).square()*known).sum()/known.sum().clamp_min(1)
   opt.zero_grad();loss.backward();opt.step()
  out=Path(a.output_base)/a.cohort/'native'/a.key;out.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/'field.pt',out/'field.pt');shutil.copyfile(parent/'calibration.pt',out/'calibration.pt')
  torch.save(dict(feature_slots={name:F.normalize(sums[name],dim=1).half().cpu() for name in modes},matching='maximum native cosine across slots',supported=supported.cpu(),geometry=binding,native_space='siglip',log_scale=float(log_scale),bias=float(bias)),out/'native_field.pt')
  torch.save(dict(online,scores=((log_scale.exp()*cosine+bias).sigmoid()*supported).detach().cpu(),selection_scores=((log_scale.exp()*cosine+bias).sigmoid()*supported).detach().cpu(),selection_geometry=binding,semantic_geometry=binding,selection_observed=supported[:,0].cpu(),semantic_observed=supported[:,0].cpu()),out/'typed_readout.pt')
  torch.save(dict(texts=online['texts'],scores=cosine.cpu()*supported.cpu(),semantic_geometry=binding,semantic_observed=supported[:,0].cpu()),out/'ranking_readout.pt')
  (out/'fit.json').write_text(json.dumps(dict(mode=mode,residual_from=a.residual_from,native_update='separate area-ordered scale slots with shared maximum-cosine calibration',slots=modes,seconds=time.monotonic()-start,native_dim=d,source_frames=source_frames,calibration_frames=info['train_frames'],fit_raster_hw=[h,w],calibration_raster_hw=[ch,cw],original_resolution_evidence_preserved=True,calibration_steps=240,old_field_targets=False,query_independent_features=True,query_conditioned_source_calibration=True,all_four_teachers_retrained=False),indent=2)+'\n')
if __name__=='__main__':main()
