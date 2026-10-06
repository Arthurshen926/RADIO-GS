"""Expanded per-frame teachers on frozen next66 geometry; fixed original calibration evidence."""
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
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--arm',default='adaptive_allrgb');p.add_argument('--cohort',choices=['old','coverage64','all'],default='all');p.add_argument('--fit-width',type=int,default=128);p.add_argument('--output-base',default='output/expanded_semantic_next72');p.add_argument('--residual-from');p.add_argument('--region-weights');p.add_argument('--pooling',choices=['mean','logmeanexp','scope_max'],required=True);a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
 parent=Path('output/unified_response_next53/direct')/a.key;data=load(parent/'field.pt');online=load(parent/'typed_readout.pt');root=Path(data['root']);gr=json.loads(Path(data['geometry_report']).read_text());gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);from radio_gs.utils.immutable_artifacts import sha256_file
 geometry_path=Path('output/joint_carrier_next66')/a.arm/a.key.removeprefix('lerf_')/'geometry.pt';binding=dict(path=str(geometry_path),sha256=sha256_file(geometry_path));geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
 info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
 ch,cw=h,w
 evidence=Path('output/source_coverage_next72')/a.key.removeprefix('lerf_');source_frames=json.loads((evidence/'cohorts.json').read_text())['frames'][a.cohort];h=round(h*a.fit_width/w);w=a.fit_width
 from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
 q=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].half().float().cuda(),dim=1)
 observed=torch.ones(len(geo.xyz),dtype=torch.bool,device='cuda');examples=[];d=q.shape[1]
 from radio_gs.querying.region_bank import query_region_bank
 manifest=Path('output/region_bank_next78')/a.key.removeprefix('lerf_')/'manifest.json'
 cosine,supported=query_region_bank(manifest,q,pooling=a.pooling)
 h,w=ch,cw
 with torch.no_grad():
  for frame in info['train_frames']:
   cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
   renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
   hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights'];op,_,valid=visible_operator(hit,observed,h*w,False)
   record=load(Path('output/typed_evidence_next44')/a.key/f'identity_{frame}.pt')
   target=torch.stack([F.interpolate(r['masks'].float().amax(0)[None,None],size=(ch,cw),mode='bilinear',align_corners=False)[0,0] if len(r['scores']) else torch.zeros(ch,cw) for r in record['instances']],-1).reshape(ch*cw,-1).cuda()
   active=torch.tensor([len(r['scores'])>0 for r in record['instances']],device='cuda');binary=(target.T.reshape(-1,1,ch,cw)>.5).float();pos=(-F.max_pool2d(-binary,3,1,1)>.5).flatten(1).T&valid;neg=(F.max_pool2d(binary,3,1,1)<.5).flatten(1).T&valid
   examples.append((op,target,pos&active,neg&active));print(a.key,frame,'features fused',flush=True)
 for mode in [a.pooling]:
  # Two scalar calibration parameters, trained only on cached source detections.
  log_scale=torch.nn.Parameter(torch.tensor(3.,device='cuda'));bias=torch.nn.Parameter(torch.tensor(-5.,device='cuda'));opt=torch.optim.Adam([log_scale,bias],lr=.03)
  for step in range(240):
   op,y,pos,neg=examples[step%len(examples)];value=(log_scale.exp()*cosine+bias).sigmoid()*supported;pred=torch.sparse.mm(op,value);known=pos|neg;loss=((pred-y).square()*known).sum()/known.sum().clamp_min(1)
   opt.zero_grad();loss.backward();opt.step()
  out=Path(a.output_base)/a.cohort/'native'/a.key;out.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/'field.pt',out/'field.pt');shutil.copyfile(parent/'calibration.pt',out/'calibration.pt')
  (out/'region_bank.json').write_text(json.dumps(dict(manifest=str(manifest),pooling=a.pooling,inverse_temperature=20.,log_scale=float(log_scale),bias=float(bias),query_vocabulary_frozen=False),indent=2)+'\n')
  torch.save(dict(online,scores=((log_scale.exp()*cosine+bias).sigmoid()*supported).detach().cpu(),selection_scores=((log_scale.exp()*cosine+bias).sigmoid()*supported).detach().cpu(),selection_geometry=binding,semantic_geometry=binding,selection_observed=supported[:,0].cpu(),semantic_observed=supported[:,0].cpu()),out/'typed_readout.pt')
  torch.save(dict(texts=online['texts'],scores=cosine.cpu()*supported.cpu(),semantic_geometry=binding,semantic_observed=supported[:,0].cpu()),out/'ranking_readout.pt')
  (out/'fit.json').write_text(json.dumps(dict(pooling=a.pooling,manifest=str(manifest),seconds=time.monotonic()-start,native_dim=d,source_frames=source_frames,calibration_frames=info['train_frames'],calibration_raster_hw=[ch,cw],query_independent_bank=True,query_conditioned_source_calibration=True,late_region_pooling=True),indent=2)+'\n')
if __name__=='__main__':main()
