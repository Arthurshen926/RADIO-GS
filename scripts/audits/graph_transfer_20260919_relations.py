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
 out=Path('output/graph_transfer_20260919')/a.scene;assert out.exists()
 registration=dict(scene=a.scene,k=16,lambda_graph=1.,unary_confidence=.5,query_gate='all native calibrated probabilities',feature_bandwidth='median positive source-only neighbor distance squared',hard_seeds=False,gt_parameter_fitting=False,source='train frames only',arms=['direct_dino','distilled_structure32'],not_exact_ludvig_reproduction=True)

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
 from scipy.stats import spearmanr
 d1=knn_feature_distances(features,neighbors,distance_chunk_size=1024).cpu().numpy();d2=knn_feature_distances(structure,neighbors,distance_chunk_size=1024).cpu().numpy();rng=np.random.default_rng(20260919);sample=rng.choice(d1.size,min(100000,d1.size),replace=False)
 top1=np.argsort(d1,axis=1)[:,:4];top2=np.argsort(d2,axis=1)[:,:4];overlap=np.array([len(set(x)&set(y))/4 for x,y in zip(top1,top2)])
 report=dict(scene=a.scene,geometry_spatial_neighbors_identical=True,edge_sample=len(sample),distance_spearman=float(spearmanr(d1.ravel()[sample],d2.ravel()[sample]).statistic),top4_neighbor_retention=float(overlap.mean()),direct_dimension=768,distilled_dimension=32,not_pure_distillation_causal_test=True,direct_feature_source='train-view DINO dense maps responsibility-weighted',distilled_feature_source='frozen jointly trained structural suffix',query_or_gt_used=False)
 (out/'relations.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':
 with torch.no_grad():main()
