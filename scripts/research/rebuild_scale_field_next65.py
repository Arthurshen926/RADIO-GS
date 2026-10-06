"""Rebuild native teacher features on the refined carrier, without old-field targets."""
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
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
 parent=Path('output/unified_response_next53/direct')/a.key;data=load(parent/'field.pt');online=load(parent/'typed_readout.pt');root=Path(data['root']);gr=json.loads(Path(data['geometry_report']).read_text());gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);binding=online['selection_geometry'];geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
 info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
 ep=load(root/'episodes.pt');masks=ep['masks'];region_views=ep['proposal_views'];del ep
 teacher=F.normalize(load(root/info['teacher_files']['siglip']['path'])['descriptors'].float().cuda(),dim=1)
 q=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].half().float().cuda(),dim=1)
 observed=torch.ones(len(geo.xyz),dtype=torch.bool,device='cuda');n=len(observed);d=teacher.shape[1];examples=[];sums={m:torch.zeros(n,d) for m in ['fine','middle','coarse']};mass={m:torch.zeros(n,1,device='cuda') for m in sums}
 with torch.no_grad():
  for frame in info['train_frames']:
   vi=info['frames'].index(frame);ids=torch.where(region_views==vi)[0];mask=masks[ids].flatten(1).float().cuda();count=mask.sum(0);desc=teacher[ids.cuda()]
   areas=mask.sum(1);ordered=areas[:,None].expand_as(mask).masked_fill(mask<=.5,float('inf')).argsort(dim=0)
   covered=(mask>.5).sum(0);pixel=torch.arange(h*w,device='cuda')
   levels={}
   for name,fraction in [('fine',0.),('middle',.5),('coarse',1.)]:
    rank=((covered-1).clamp_min(0).float()*fraction).long()
    levels[name]=desc[ordered[rank,pixel]]
   cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
   renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
   hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights'];op,_,valid=visible_operator(hit,observed,h*w,False)
   pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();weight=hit['weights'].float().cuda()*(count[pid]>0)
   adj=torch.sparse_coo_tensor(torch.stack([gid,pid]),weight,(n,h*w)).coalesce()
   for mode,grid in levels.items():
    for channel in range(0,d,128):sums[mode][:,channel:channel+128]+=torch.sparse.mm(adj,grid[:,channel:channel+128].contiguous()).cpu()
    mass[mode]+=torch.sparse.mm(adj,torch.ones(h*w,1,device='cuda'))
   record=load(Path('output/typed_evidence_next44')/a.key/f'identity_{frame}.pt')
   target=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in record['instances']],-1).reshape(h*w,-1).cuda()
   active=torch.tensor([len(r['scores'])>0 for r in record['instances']],device='cuda');binary=(target.T.reshape(-1,1,h,w)>.5).float();pos=(-F.max_pool2d(-binary,3,1,1)>.5).flatten(1).T&valid;neg=(F.max_pool2d(binary,3,1,1)<.5).flatten(1).T&valid
   examples.append((op,target,pos&active,neg&active));print(a.key,frame,'features fused',flush=True)
 slot_cosines=[]
 for mode in sums:
  features=F.normalize(sums[mode],dim=1)
  slot_cosines.append(torch.cat([part.cuda()@q.T for part in features.split(8192)]))
  out=Path('output/scale_field_next65/native')/a.key;out.mkdir(parents=True,exist_ok=True)
  torch.save(dict(features=features.half(),supported=(mass[mode]>0).cpu(),geometry=binding,native_space='siglip',scale=mode),out/(mode+'_field.pt'))
 cosine=torch.stack(slot_cosines).amax(0).detach()
 for mode in ['native']:
  supported=mass['fine']>0
  # Two scalar calibration parameters, trained only on cached source detections.
  log_scale=torch.nn.Parameter(torch.tensor(3.,device='cuda'));bias=torch.nn.Parameter(torch.tensor(-5.,device='cuda'));opt=torch.optim.Adam([log_scale,bias],lr=.03)
  for step in range(240):
   op,y,pos,neg=examples[step%len(examples)];value=(log_scale.exp()*cosine+bias).sigmoid()*supported;pred=torch.sparse.mm(op,value);known=pos|neg;loss=((pred-y).square()*known).sum()/known.sum().clamp_min(1)
   opt.zero_grad();loss.backward();opt.step()
  out=Path('output/scale_field_next65')/mode/a.key;out.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/'field.pt',out/'field.pt');shutil.copyfile(parent/'calibration.pt',out/'calibration.pt')
  torch.save(dict(slots=['fine_field.pt','middle_field.pt','coarse_field.pt'],readout='max native cosine across region-area ranks',log_scale=float(log_scale),bias=float(bias),geometry=binding),out/'native_field.pt')
  torch.save(dict(online,scores=((log_scale.exp()*cosine+bias).sigmoid()*supported).detach().cpu()),out/'typed_readout.pt')
  torch.save(dict(online,scores=cosine.cpu()*supported.cpu()),out/'ranking_readout.pt')
  (out/'fit.json').write_text(json.dumps(dict(mode=mode,seconds=time.monotonic()-start,native_dim=d,source_frames=info['train_frames'],calibration_steps=240,old_field_targets=False,query_independent_features=True,feature_slots=3,no_query_dependent_slot_training=True,query_conditioned_source_calibration=True,all_four_teachers_retrained=False),indent=2)+'\n')
if __name__=='__main__':main()
