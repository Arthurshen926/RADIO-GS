"""Attach existing source identity/extent inference to the rebuilt native field."""
import argparse,json,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.querying.typed_evidence import visible_operator,view_evidence,solve_set,SetType,ViewEvidence
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--arm',choices=['abstain','positive'],required=True);a=p.parse_args();torch.set_num_threads(4)
 parent=Path('output/native_rebuild_next54/mixture')/a.key;data=load(parent/'field.pt');online=load(parent/'typed_readout.pt');prior=online['scores'].cuda();root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];b=online['selection_geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda');gr=json.loads(Path(data['geometry_report']).read_text());gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);observed=torch.ones(len(prior),device='cuda',dtype=torch.bool)
 records={f:load(Path('output/typed_evidence_next44')/a.key/f'identity_{f}.pt') for f in info['train_frames']};square=torch.zeros(prior.shape[1]);count=torch.zeros_like(square)
 for d in records.values():
  for j,r in enumerate(d['instances']):square[j]+=r['foreground_margin'].square().sum();count[j]+=len(r['scores'])
 scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6);views=[]
 tracks=[load(Path('output/text_video_next63/tracks')/a.key/f'{qi}.pt') for qi in range(prior.shape[1])]
 seed_confidence=[]
 for qi,t in enumerate(tracks):
  r=records[t['seed_frame']]['instances'][qi] if t['seed_frame'] is not None else None
  seed_confidence.append(float((r['scores']*(r['foreground_margin']/scale[qi]).sigmoid()).max()) if r is not None else 0.)
 evidence=[]
 for f,d in records.items():
  cam=gp.cameras[f];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
  hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights'];op,eop,valid=visible_operator(hit,observed,h*w,False)
  target=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in d['instances']],-1).reshape(h*w,-1).cuda();active=torch.tensor([len(r['scores'])>0 for r in d['instances']],device='cuda');confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(d['instances'])],device='cuda')
  present=torch.tensor([bool(t['frames'].get(f,{}).get('present',False)) and f!=t['seed_frame'] for t in tracks],device='cuda')
  temporal=torch.stack([t['frames'][f]['probability'].float() if bool(present[qi]) else torch.zeros(h,w) for qi,t in enumerate(tracks)],-1).reshape(h*w,-1).cuda()
  original=view_evidence(op,eop,valid,target,active,f,h,w,confidence)
  tracked=view_evidence(op,eop,valid,temporal,present,f,h,w,torch.tensor(seed_confidence,device='cuda'))
  # Temporal presence can invalidate a detector exclusion. It cannot certify
  # the complement of a partial tracked instance as negative for a text union.
  conflict=original.negative&tracked.positive
  negative=original.negative&~tracked.positive
  extra=tracked.positive&~original.positive if a.arm=='positive' else torch.zeros_like(tracked.positive)
  positive=original.positive|extra
  y=torch.where(extra,temporal,target)
  weight=torch.where(extra,tracked.confidence,original.confidence).expand_as(target)
  visible=original.visible|extra
  views.append(ViewEvidence(op,eop.T.coalesce(),eop,y,positive,negative,visible,f,weight))
  evidence.append(dict(frame=f,conflicting_entries=int(conflict.sum()),added_positive_entries=int(extra.sum()),original_positive_entries=int(original.positive.sum()),original_negative_entries=int(original.negative.sum())))
 result,stats=solve_set(prior,views,SetType.TEXT_UNION,corroborate=False,consistency=True,steps=60)
 out=Path('output/typed_temporal_next64')/(a.arm+'_scope')/a.key;out.mkdir(parents=True,exist_ok=True)
 for name in ['field.pt','calibration.pt']:shutil.copyfile(parent/name,out/name)
 torch.save(dict(online,scores=result.cpu()),out/'typed_readout.pt');(out/'fit.json').write_text(json.dumps(dict(stats,arm=a.arm,evidence=evidence,temporal_background_is_unknown=True,source_only=True,native_parent=str(parent),selection_unchanged=True),indent=2)+'\n')
if __name__=='__main__':main()
