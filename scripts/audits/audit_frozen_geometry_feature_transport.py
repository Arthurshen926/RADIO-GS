"""Synthetic feature invariants on real frozen geometry, no semantic training."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from radio_gs.scripts.repair_moge_geometry import load_views
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);device='cuda:0';records=[]
for scene in ('figurines','ramen','teatime','waldo_kitchen'):
 report=json.loads((a.root/scene/'report.json').read_text());g=report['results']['independent_scale']
 model=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=device)
 binding=model.fingerprint()
 fusion=Path('/mnt/pool/sqy/results/RADIO-GS/output/v4_fusion_contract_repair_20260914')/(scene+'.json')
 prior=MogeGeometryPrior(report.get('source_report',str(fusion)),report['source_report_sha256'])
 cam=prior.cameras[report['heldout_frames'][0]];h,w=120,160
 k=cam.intrinsic.clone();k[0]*=w/cam.width;k[1]*=h/cam.height
 renderer=FeatureFieldRenderer(h,w,float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2]),max_channels_per_chunk=2).to(device)
 features=torch.cat([torch.ones(model.xyz.shape[0],1,device=device),model.xyz],dim=1)
 pose=torch.linalg.inv(cam.camera_to_world).float().to(device)
 result=renderer.render_feature_rows(model,pose,features,alpha_normalize=True)
 visible=result['alpha_map']>.01
 constant_error=float((result['feature_map'][0][visible]-1).abs().max())
 query=torch.tensor([.1,.2,.3,.4],device=device)
 scalar=renderer.render_feature_rows(model,pose,features@query[:,None],alpha_normalize=True)
 before=(result['feature_map']*query[:,None,None]).sum(0)
 linear_error=float((before[visible]-scalar['feature_map'][0][visible]).abs().max())
 assert constant_error<2e-5 and linear_error<2e-4
 assert torch.allclose(result['depth_map'],scalar['depth_map'],atol=1e-6)
 assert torch.equal(result['alpha_map'],scalar['alpha_map'])
 confidence=torch.linspace(0.,1.,len(features),device=device)
 weighted=renderer.render_feature_rows(model,pose,features,alpha_normalize=True,row_confidence=confidence)
 assert torch.equal(result['alpha_map'],weighted['alpha_map'])
 assert torch.allclose(result['depth_map'],weighted['depth_map'],atol=1e-6)
 support=weighted['feature_support_map']>1e-5
 assert float((weighted['feature_map'][0][support]-1).abs().max())<2e-5
 weighted_scalar=renderer.render_feature_rows(model,pose,features@query[:,None],alpha_normalize=True,row_confidence=confidence)
 confidence_linear_error=float(((weighted['feature_map']*query[:,None,None]).sum(0)[support]-weighted_scalar['feature_map'][0][support]).abs().max())
 assert confidence_linear_error<2e-4
 model.assert_binding(binding)
 records.append({'scene':scene,'constant_field_max_error':constant_error,'linear_2d_3d_commutation_max_error':linear_error,
                 'confidence_preserves_alpha_depth':True,'confidence_linear_error':confidence_linear_error,
                 'geometry_alpha_depth_unchanged_by_features':True,'full_geometry_binding_unchanged':True,
                 'geometry_sha256':g['geometry_sha256'],'role':'transport_correctness_not_semantic_accuracy'})
 del model,features,result,scalar;torch.cuda.empty_cache()
a.output.write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
