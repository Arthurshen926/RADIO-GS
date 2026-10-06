"""Source-response reconstruction on the new carrier; not four-teacher retraining."""
import argparse,json,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.querying.typed_evidence import visible_operator
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.utils.immutable_artifacts import sha256_file

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();torch.set_num_threads(4)
 parent=Path('output/depth_selection_next52_recovered/fit')/a.key;data=load(parent/'field.pt');online=load(parent/'typed_readout.pt');root=Path(data['root']);observed=load(data['base_field'])['state_dict']['observed'].cuda();prior=online['scores'].cuda()
 gr=json.loads(Path(data['geometry_report']).read_text());gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);binding=online['selection_geometry'];geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
 info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];views=[];den=torch.zeros(len(prior),1,device='cuda')
 for f in info['train_frames']:
  old=load(root/auth['views'][info['frames'].index(f)]['relative_path']);op0,_,valid0=visible_operator(old,observed,h*w,False);target=torch.sparse.mm(op0,prior)
  cam=gp.cameras[f];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
  renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
  hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights'];op,_,valid=visible_operator(hit,observed,h*w,False);known=(valid&valid0).float();adj=op.T.coalesce();den+=torch.sparse.mm(adj,known);views.append((op,adj,target,known))
 def error(x):return float(sum(((torch.sparse.mm(op,x)-t).square()*v).sum() for op,ad,t,v in views)/(sum(v.sum() for op,ad,t,v in views)*x.shape[1]).clamp_min(1))
 before=error(prior)
 for mode in ['direct','transport']:
  value=prior.clone()
  if mode=='transport':
   for step in range(120):
    gradient=sum(torch.sparse.mm(ad,(t-torch.sparse.mm(op,value))*v) for op,ad,t,v in views)
    value=(value+.95*gradient/den.clamp_min(1e-8)).clamp(0,1)
  out=Path('output/unified_response_next53')/mode/a.key;out.mkdir(parents=True,exist_ok=True)
  report=json.loads(json.dumps(gr));report['results']={'independent_scale':dict(geometry=binding['path'],geometry_sha256=binding['sha256'])};report['research_intervention_only']=True;rp=out/'geometry_report.json';rp.write_text(json.dumps(report,indent=2)+'\n')
  # Copy legacy metadata only for the frozen evaluation interface. No old
  # field or source caches may be used to claim a newly trained semantic map.
  payload=dict(data,geometry_report=str(rp),geometry_report_sha256=sha256_file(rp));torch.save(payload,out/'field.pt');shutil.copyfile(parent/'calibration.pt',out/'calibration.pt')
  result=dict(online,scores=value.cpu(),readout_contract='source_response_transport_same_carrier');torch.save(result,out/'typed_readout.pt')
  (out/'fit.json').write_text(json.dumps(dict(mode=mode,source_only=True,query_response_distillation=True,full_feature_field_retrained=False,iterations=120 if mode=='transport' else 0,source_mse_before=before,source_mse_after=error(value),geometry=binding),indent=2)+'\n')
  print(a.key,mode,before,error(value),flush=True)
if __name__=='__main__':main()
