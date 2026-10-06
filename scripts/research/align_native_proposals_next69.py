"""Source-visibility weighted confidence transport; no target labels or images."""
import argparse,json,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.querying.typed_evidence import visible_operator
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--arm',required=True);a=p.parse_args();torch.set_num_threads(4)
 parent=Path('output/native_proposals_next69/scope')/a.key;data=load(parent/'field.pt');online=load(parent/'typed_readout.pt');candidate=online['scores'].cuda();prior=load(Path('output/unified_response_next53/direct')/a.key/'typed_readout.pt')['scores'].cuda();observed=load(data['base_field'])['state_dict']['observed'].cuda();root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];b=online['selection_geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda');gr=json.loads(Path(data['geometry_report']).read_text());gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);old_geo=FrozenGaussianGeometry.load(load(Path('output/unified_response_next53/direct')/a.key/'typed_readout.pt')['selection_geometry']['path'],expected_sha256=load(Path('output/unified_response_next53/direct')/a.key/'typed_readout.pt')['selection_geometry']['sha256'],device='cuda');new_observed=online['semantic_observed'].cuda();mass=torch.zeros(len(candidate),device='cuda');old_mass=torch.zeros(len(prior),device='cuda')
 for f in info['train_frames']:
  cam=gp.cameras[f];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
  hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights'];op,_,valid=visible_operator(hit,new_observed,h*w,False);mass+=torch.sparse.mm(op.T.coalesce(),valid.float())[:,0]
  old_hit=rasterize_single_view_contributions(old_geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);old_hit['base_weights']=old_hit['weights'];old_op,_,old_valid=visible_operator(old_hit,observed,h*w,False);old_mass+=torch.sparse.mm(old_op.T.coalesce(),old_valid.float())[:,0]
 ids=torch.where(mass>1e-8)[0];old_ids=torch.where(old_mass>1e-8)[0];weight=mass[ids];old_weight=old_mass[old_ids];value=candidate.clone()
 for j in range(prior.shape[1]):
  x=candidate[ids,j];y=prior[old_ids,j];ix=torch.argsort(x,stable=True);iy=torch.argsort(y,stable=True)
  wx=weight[ix];wy=old_weight[iy];cx=(wx.cumsum(0)-.5*wx)/wx.sum();cy=(wy.cumsum(0)-.5*wy)/wy.sum()
  index=torch.searchsorted(cy,cx).clamp_max(len(cy)-1);value[ids[ix],j]=y[iy[index]]
 out=Path('output/native_proposals_next69/semantic')/a.key;out.mkdir(parents=True,exist_ok=True)
 for name in ['field.pt','calibration.pt']:shutil.copyfile(parent/name,out/name)
 retained=load(Path('output/subset_consistency_next67/exposed')/a.key/'typed_readout.pt');torch.save(dict(online,scores=value.cpu(),selection_scores=retained['selection_scores']),out/'typed_readout.pt');(out/'fit.json').write_text(json.dumps(dict(source_only=True,prior='next53 direct confidence distribution',candidate='native mixture plus detector and native region source scope',spatial_assignment='candidate ordering',weight='source visible pixel contribution',unobserved='retain rebuilt candidate',selection_unchanged=True,confidence_transport_not_new_features=True),indent=2)+'\n')
if __name__=='__main__':main()
