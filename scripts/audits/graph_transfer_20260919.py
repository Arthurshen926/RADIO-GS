"""Registered soft-unary graph control, not exact LUDVIG reproduction."""
import argparse,json,sys,runpy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial import cKDTree
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.querying.query_conditioned_diffusion import QueryConditionedDiffusionConfig,solve_continuous_query_support,knn_feature_distances

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
 out=Path('output/graph_transfer_20260919')/a.scene;out.mkdir(parents=True,exist_ok=False)
 registration=dict(scene=a.scene,k=16,lambda_graph=1.,unary_confidence=.5,query_gate='all native calibrated probabilities',feature_bandwidth='median positive source-only neighbor distance squared',hard_seeds=False,gt_parameter_fitting=False,source='train frames only',arms=['direct_dino','distilled_structure32'],not_exact_ludvig_reproduction=True)
 (out/'registration.json').write_text(json.dumps(registration,indent=2))
 root=Path('output/depth_opacity_pipeline_20260918')/a.scene/'reconstruction/field';gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json';gr=json.loads(gp.read_text());g=gr['results']['independent_scale']
 from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
 geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0')
 payload=load(root/'bounded_fallback_admission/field.pt');geo.assert_binding(payload['geometry_binding']);sd=payload['state_dict'];observed=sd['observed'];n=len(observed)
 data=load(Path('output/lerf_native_membership_20260919')/a.scene/'scores.pt');data['scores']={k:data['scores'][k] for k in ['native_compact_calibrated','final']}
 ids=torch.where(observed)[0];xyz=geo.xyz.cpu().numpy()[ids];dist,neigh=cKDTree(xyz).query(xyz,k=17,workers=4);neighbors=torch.from_numpy(neigh[:,1:].copy()).long().cuda();spatial=torch.from_numpy(dist[:,1:].copy()).float().cuda();spatial_scale=spatial[:,-1].median().clamp_min(1e-8)
 info=json.loads((root/'teachers.json').read_text());rec=info['teacher_files']['dino'];assert sha256_file(root/rec['path'])==rec['sha256'];teacher=load(root/rec['path']);ep=load(root/'episodes.pt');h,w=ep['masks'].shape[-2:];auth=json.loads((root/'responsibility.json').read_text());features=torch.zeros(n,teacher.shape[1],device='cuda');mass=torch.zeros(n,device='cuda')
 for frame in info['train_frames']:
  vi=info['frames'].index(frame);hit=load(root/auth['views'][vi]['relative_path']);pix=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();weights=hit['base_weights'].float().cuda();dense=F.interpolate(teacher[vi:vi+1].float().cuda(),size=(h,w),mode='bilinear',align_corners=False)[0].flatten(1).T;dense=F.normalize(dense,dim=1)
  mat=torch.sparse_coo_tensor(torch.stack([gid,pix]),weights,(n,h*w)).coalesce();features+=torch.sparse.mm(mat,dense);mass.index_add_(0,gid,weights)
  print(a.scene,'lift',frame,flush=True)
 features=F.normalize(features[ids.cuda()]/mass[ids.cuda(),None].clamp_min(1e-8),dim=1);structure=F.normalize(sd['latent'][ids,64:].float().cuda(),dim=1)
 cfg=QueryConditionedDiffusionConfig(kernel='continuous_convex_v2',edge_binarize_threshold=None,laplacian_weight=1.,cg_iterations=64,hard_observation_confidence=.99)
 diagnostics={}
 for arm,f in [('direct_dino',features),('distilled_structure32',structure)]:
  dd=knn_feature_distances(f,neighbors,distance_chunk_size=1024).square();scale=dd[dd>0].median().clamp_min(1e-8);aff=torch.exp(-dd/scale)*torch.exp(-spatial.square()/(2*spatial_scale.square()));diagnostics[arm]=dict(feature_scale=float(scale),mean_affinity=float(aff.mean()))
  result=torch.zeros_like(data['scores']['native_compact_calibrated']);confidence=torch.full((len(ids),),.5,device='cuda')
  for qi,text in enumerate(data['texts']):
   unary=data['scores']['native_compact_calibrated'][ids,qi].cuda();sol=solve_continuous_query_support(unary,confidence,neighbors,aff,unary,config=cfg);result[ids,qi]=sol.cpu();print(a.scene,arm,qi,text,flush=True)
  data['scores'][arm+'_calibrated']=result
 torch.save(data,out/'scores.pt');prep=dict(scene=a.scene,geometry_report_sha256=sha256_file(gp),scores_sha256=sha256_file(out/'scores.pt'),registration=registration,diagnostics=diagnostics,source_teacher_sha256=rec['sha256'],implementation_sha256=sha256_file(__file__),target_rgb_used=False,geometry_unchanged=True)
 (out/'preparation.json').write_text(json.dumps(prep,indent=2))
 # Reuse exact frozen-grid renderer and baseline pixel replay checks.
 sys.argv=['diagnose_lerf_native_membership.py','--scene',a.scene,'--output',str(out),'--render-only','--device','cuda:0'];runpy.run_path('scripts/audits/diagnose_lerf_native_membership.py',run_name='__main__')
if __name__=='__main__':
 with torch.no_grad():main()
