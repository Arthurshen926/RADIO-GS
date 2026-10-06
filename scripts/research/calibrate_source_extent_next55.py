"""Fit a shared monotone source-evidence calibration without old confidence mass."""
import argparse,json,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.querying.typed_evidence import visible_operator,view_evidence,solve_set,SetType
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--domain',choices=['pixel','element'],default='pixel');a=p.parse_args();torch.set_num_threads(4)
 parent=Path('output/native_rebuild_next54/mixture_scope')/a.key;data=load(parent/'field.pt');online=load(parent/'typed_readout.pt');prior=online['scores'].cuda();root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];b=online['selection_geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda');gr=json.loads(Path(data['geometry_report']).read_text());gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);observed=torch.ones(len(prior),device='cuda',dtype=torch.bool)
 records={f:load(Path('output/typed_evidence_next44')/a.key/f'identity_{f}.pt') for f in info['train_frames']};square=torch.zeros(prior.shape[1]);count=torch.zeros_like(square)
 for d in records.values():
  for j,r in enumerate(d['instances']):square[j]+=r['foreground_margin'].square().sum();count[j]+=len(r['scores'])
 scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6);views=[]
 for f,d in records.items():
  cam=gp.cameras[f];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
  hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights'];op,eop,valid=visible_operator(hit,observed,h*w,False)
  target=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in d['instances']],-1).reshape(h*w,-1).cuda();active=torch.tensor([len(r['scores'])>0 for r in d['instances']],device='cuda');confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(d['instances'])],device='cuda')
  views.append(view_evidence(op,eop,valid,target,active,f,h,w,confidence))
 # Fit one shared monotone map to source-rendered responses. Missing detections
 # and uncertain boundaries abstain; confidence weights remain source-derived.
 bins=256;total=torch.zeros(bins,device='cuda');positive=total.clone()
 for e in views:
  response=torch.sparse.mm(e.operator,prior).clamp(0,1)
  weight=(e.positive|e.negative).float()*e.confidence
  # Each detected query-view contributes equally before detector confidence.
  weight=weight/weight.sum(0,keepdim=True).clamp_min(1e-8)*e.confidence
  if a.domain=='element':
   numerator=torch.sparse.mm(e.adjoint,weight*e.positive)
   weight=torch.sparse.mm(e.adjoint,weight)
   response=prior.clamp(0,1)
  else:numerator=weight*e.positive
  index=(response*(bins-1)).long()
  total.scatter_add_(0,index.flatten(),weight.flatten())
  positive.scatter_add_(0,index.flatten(),numerator.flatten())
 # Weighted pool-adjacent-violators fit; no target labels and no old response.
 blocks=[]
 for i,(n,y) in enumerate(zip(total.cpu().tolist(),positive.cpu().tolist())):
  if n==0:continue
  blocks.append([i,i,n,y])
  while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
   right=blocks.pop();left=blocks.pop()
   blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
 knots=torch.tensor([(x[0]+x[1])*.5/(bins-1) for x in blocks],device='cuda')
 values=torch.tensor([x[3]/x[2] for x in blocks],device='cuda')
 def transform(x):
  hi=torch.searchsorted(knots,x.contiguous()).clamp(0,len(knots)-1);lo=(hi-1).clamp_min(0)
  alpha=((x-knots[lo])/(knots[hi]-knots[lo]).clamp_min(1e-8)).clamp(0,1)
  return values[lo]+alpha*(values[hi]-values[lo])
 result=transform(prior)
 retained=load(Path('output/native_rebuild_next54/mixture_masscal')/a.key/'typed_readout.pt')['scores'].cuda()
 diagnostics={}
 for name,field in [('uncalibrated',prior),('source_calibrated',result),('retained_old_mass',retained)]:
  error=den=tp=pos=tn=neg=0.
  for e in views:
   rendered=torch.sparse.mm(e.operator,field)
   known=(e.positive|e.negative).float()*e.confidence
   error+=float(((rendered-e.target).square()*known).sum());den+=float(known.sum())
   tp+=float(((rendered>.5)*e.positive*e.confidence).sum());pos+=float((e.positive*e.confidence).sum())
   tn+=float(((rendered<=.5)*e.negative*e.confidence).sum());neg+=float((e.negative*e.confidence).sum())
  diagnostics[name]=dict(source_brier=error/max(den,1e-8),source_positive_recall=tp/max(pos,1e-8),source_negative_specificity=tn/max(neg,1e-8))
 stats=dict(source_fit_diagnostics=diagnostics,blocks=blocks,shared_across_queries=True,old_confidence_mass=False,
            domain=a.domain,source_views=len(views),source_evidence='eroded positives and dilated-complement pseudo-negatives')
 out=Path('output/source_extent_next55')/('isotonic' if a.domain=='pixel' else 'element_isotonic')/a.key;out.mkdir(parents=True,exist_ok=True)
 for name in ['field.pt','calibration.pt']:shutil.copyfile(parent/name,out/name)
 torch.save(dict(online,scores=result.cpu()),out/'typed_readout.pt');(out/'fit.json').write_text(json.dumps(dict(stats,source_only=True,native_parent=str(parent),selection_unchanged=True),indent=2)+'\n')
if __name__=='__main__':main()
