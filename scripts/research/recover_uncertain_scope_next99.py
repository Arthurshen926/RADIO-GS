"""Resolve caption abstention using other-view accepted source support."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import label
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.querying.typed_evidence import visible_operator
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.utils.immutable_artifacts import sha256_file

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene;props=json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text());ids=[r['query_index'] for r in props];tracks=[load(Path('output/category_identity_tracks_next98')/key/f'{qi}.pt') for qi in ids];judgments=[json.loads((Path('output/category_identity_tracks_next98')/key/f'{qi}.json').read_text()) for qi in ids]
 base=Path('output/native_write_next75/all/calibrated')/key;field=load(base/'field.pt');initial=load(base/'typed_readout.pt');observed=initial['semantic_observed'].cuda();gr=json.loads(Path(field['geometry_report']).read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);path=Path('output/joint_carrier_next66/adaptive_allrgb')/a.scene/'geometry.pt';geo=FrozenGaussianGeometry.load(path,expected_sha256=sha256_file(path),device='cuda');frames=json.loads((Path('output/source_coverage_next72')/a.scene/'cohorts.json').read_text())['frames']['all'];h,w=192,256
 def operator(frame):
  cam=prior.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*prior.scale,far_plane=100*prior.scale);hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights'];op,_,_=visible_operator(hit,observed,h*w,False);return op
 votes=torch.zeros((len(observed),len(ids)),device='cuda');own={};byframe=[{} for _ in ids]
 for qi,d in enumerate(judgments):
  for j in d['judgments']:byframe[qi].setdefault(j['frame'],[]).append(j)
 for frame in frames:
  masks=[]
  for t in tracks:
   r=t['frames'].get(frame,{})
   if not r.get('present',False) or frame==t['seed_frame']:masks.append(torch.zeros(h,w));continue
   binary=(r['probability'].float()>.5).float();core=-F.max_pool2d(-binary[None,None],3,1,1)[0,0]>.5;masks.append((core&r.get('positive_allowed',torch.zeros(h,w,dtype=torch.bool))).float())
  positive=torch.stack(masks,-1).reshape(h*w,-1).cuda()
  if not positive.any():continue
  op=operator(frame);mass=torch.sparse.sum(op,dim=0).to_dense();purity=torch.sparse.mm(op.T,positive)/mass[:,None].clamp_min(1e-8);witness=(purity>.5)&(mass[:,None]>1e-8);votes+=witness;own[frame]=[torch.where(witness[:,i])[0].cpu() for i in range(len(ids))]
 decisions=[[] for _ in ids]
 for frame in frames:
  if not any(any(j['answer']=='C' for j in bf.get(frame,[])) for bf in byframe):continue
  other=votes.clone()
  for i,indices in enumerate(own.get(frame,[[] for _ in ids])):other[indices,i]-=1
  op=operator(frame);response=torch.sparse.mm(op,(other>=2).float()).reshape(h,w,-1)
  for i,t in enumerate(tracks):
   js=byframe[i].get(frame,[])
   if not js:continue
   labels,_=label((t['frames'][frame]['probability']>.5).numpy(),structure=np.ones((3,3)))
   for j in js:
    if j['answer']!='C':continue
    mask=torch.from_numpy(labels==j['component']).cuda();fraction=float(response[:,:,i][mask].mean());decisions[i].append(dict(frame=frame,component=j['component'],other_view_support=fraction,accepted=fraction>.5))
 for arm in ['all','geometry']:
  out=Path('output/uncertain_tracks_next99')/arm/key;out.mkdir(parents=True,exist_ok=True)
  for i,qi in enumerate(ids):
   t=load(Path('output/category_identity_tracks_next98')/key/f'{qi}.pt');recovered={(r['frame'],r['component']) for r in decisions[i] if r['accepted']};counts=dict(unknown=0,recovered=0)
   for frame,js in byframe[i].items():
    r=t['frames'][frame];labels,_=label((r['probability']>.5).numpy(),structure=np.ones((3,3)));accepted=[]
    for j in js:
     if j['answer']=='C':counts['unknown']+=1
     take=j['answer']=='A' or (j['answer']=='C' and (arm=='all' or (frame,j['component']) in recovered))
     if take:accepted.append(j['component']);counts['recovered']+=int(j['answer']=='C')
    allow=np.isin(labels,accepted);r.update(positive_allowed=torch.from_numpy(allow),negative_allowed=torch.ones(allow.shape,dtype=torch.bool),scope_reliability=float(allow.any()))
   torch.save(t,out/f'{qi}.pt');(out/f'{qi}.json').write_text(json.dumps(dict(query=t['query'],query_index=qi,arm=arm,counts=counts,geometry_decisions=decisions[i],source_only=True,source_raster=[h,w],other_accepted_view_witnesses=2,primitive_positive_purity=.5,component_projected_support=.5,recursive_admission=False),indent=2)+'\n');print(key,arm,t['query'],counts,flush=True)
if __name__=='__main__':main()
