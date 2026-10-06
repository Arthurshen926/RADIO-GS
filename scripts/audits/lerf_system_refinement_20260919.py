"""Bounded source-region relation controls; no object-negative invention."""
import argparse,json,sys,runpy
from pathlib import Path
import torch
import numpy as np
from scipy.spatial import cKDTree
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.querying.query_conditioned_diffusion import QueryConditionedDiffusionConfig,solve_continuous_query_support,knn_feature_distances
from radio_gs.utils.immutable_artifacts import sha256_file

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def solve(scores,ids,neighbors,aff):
 cfg=QueryConditionedDiffusionConfig(kernel='continuous_convex_v2',edge_binarize_threshold=None,laplacian_weight=1.,cg_iterations=64,hard_observation_confidence=.99);result=torch.zeros_like(scores,device='cpu');confidence=torch.full((len(ids),),.5,device='cuda')
 for qi in range(scores.shape[1]):
  unary=scores[ids,qi].cuda();result[ids,qi]=solve_continuous_query_support(unary,confidence,neighbors,aff,unary,config=cfg).cpu()
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);out=Path('output/lerf_system_refinement_20260919')/a.scene;out.mkdir(parents=True,exist_ok=False)
 reg=dict(scene=a.scene,source_only=True,arms=['boundary_soft','source_rerank32'],same_graph_lambda=1.,neighbors_retained=16,candidates=32,mask_weight='inverse pixel area',minimum_informative_views=2,local_mask_max_fraction=.25,local_membership_min=.5,visibility_min='median positive contribution mass per source view',conflicting_low_high_views='cos<.1 and cos>.5 abstain',coarse_containment_not_negative=True,query_parameters_fit=False,target_gt_parameter_fit=False)
 (out/'registration.json').write_text(json.dumps(reg,indent=2))
 root=Path('output/depth_opacity_pipeline_20260918')/a.scene/'reconstruction/field';gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json';gr=json.loads(gp.read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0');payload=load(root/'bounded_fallback_admission/field.pt');geo.assert_binding(payload['geometry_binding']);sd=payload['state_dict'];observed=sd['observed'];ids=torch.where(observed)[0];n=len(observed)
 xyz=geo.xyz.cpu().numpy()[ids];tree=cKDTree(xyz);dist,nn=tree.query(xyz,k=33,workers=4);d16,n16=tree.query(xyz,k=17,workers=4)
 for row in range(len(ids)):
  base=set(n16[row,1:]);extras=[j for j in range(1,33) if nn[row,j] not in base and nn[row,j]!=row][:16];nn[row,1:]=np.r_[n16[row,1:],nn[row,extras]];dist[row,1:]=np.r_[d16[row,1:],dist[row,extras]]
 neighbors=torch.from_numpy(nn[:,1:].copy()).long().cuda();spatial=torch.from_numpy(dist[:,1:].copy()).float().cuda();ss=spatial[:,15].median().clamp_min(1e-8);features=F.normalize(sd['latent'][ids,64:].cuda(),dim=1);dd=knn_feature_distances(features,neighbors,distance_chunk_size=1024).square();fs=dd[:,:16][dd[:,:16]>0].median().clamp_min(1e-8);aff=torch.exp(-dd/fs)*torch.exp(-spatial.square()/(2*ss.square()))
 info=json.loads((root/'teachers.json').read_text());ep=load(root/'episodes.pt');auth=json.loads((root/'responsibility.json').read_text());total=torch.zeros_like(aff);count=torch.zeros_like(aff);low=torch.zeros_like(aff,dtype=torch.bool);high=low.clone();viewstats=[]
 for frame in info['train_frames']:
  vi=info['frames'].index(frame);region_ids=torch.where(ep['proposal_views']==vi)[0];masks=ep['masks'][region_ids].flatten(1).cuda().float();areas=masks.sum(1);hit=load(root/auth['views'][vi]['relative_path']);assert sha256_file(root/auth['views'][vi]['relative_path'])==auth['views'][vi]['sha256']
  lift,mass=source_pixel_evidence_weights(masks,hit['pixel_ids'].long().cuda(),hit['gaussian_ids'].long().cuda(),hit['base_weights'].float().cuda(),n,partition_overlap=False);prob=(lift[:,ids.cuda()]/mass[ids.cuda()][None].clamp_min(1e-8)).T;visibility=mass[ids.cuda()];threshold=visibility[visibility>1e-6].median();local=areas<=.25*masks.shape[1];valid=(visibility>=threshold)&((prob[:,local]>.5).any(1) if local.any() else torch.zeros(len(ids),device='cuda',dtype=torch.bool));signature=F.normalize(prob/areas.sqrt().clamp_min(1)[None],dim=1);valid_edges=0
  for start in range(0,len(ids),512):
   stop=min(start+512,len(ids));nb=neighbors[start:stop];sim=(signature[start:stop,None]*signature[nb]).sum(-1).clamp(0,1);ok=valid[start:stop,None]&valid[nb];total[start:stop]+=sim*ok;count[start:stop]+=ok;low[start:stop]|=ok&(sim<.1);high[start:stop]|=ok&(sim>.5);valid_edges+=int(ok.sum())
  viewstats.append(dict(frame=frame,visibility_median=float(threshold),informative_nodes=int(valid.sum()),informative_edges=valid_edges));print(a.scene,'source',frame,valid_edges,flush=True)
  del lift,prob,signature
 conflict=low&high;known=(count>=2)&~conflict;compat=torch.where(known,total/count.clamp_min(1),1.);gated=aff*compat;best=torch.topk(gated,16,dim=1,sorted=True).indices;graphs={'baseline':(neighbors[:,:16],aff[:,:16]),'boundary_soft':(neighbors[:,:16],gated[:,:16]),'source_rerank32':(neighbors.gather(1,best),gated.gather(1,best))}
 stats=dict(registered=reg,views=viewstats,source_known_edge_fraction=float(known.float().mean()),source_conflict_abstain_fraction=float(conflict.float().mean()),baseline_edges_halved_fraction=float((compat[:,:16]<.5).float().mean()),new_edge_fraction=float((best>=16).float().mean()),new_mean_distance=float(spatial.gather(1,best).mean()),old_mean_distance=float(spatial[:,:16].mean()),no_hard_negative_nodes=True)
 (out/'relations.json').write_text(json.dumps(stats,indent=2));torch.save(dict(ids=ids,graphs={k:(x.cpu(),y.cpu())for k,(x,y)in graphs.items()},geometry_binding=payload['geometry_binding']),out/'graphs.pt')
 bank=load(Path('output/bounded_fallback_repair_20260918')/a.scene/'text_online.encodings.pt')[0];q=F.normalize(bank['encoding'].half().float(),dim=1);model=ElementRetrievalField(**payload['architecture']);model.load_state_dict(sd);model.requires_grad_(False);final=model.retrieve(q,'text')['scores'];scores={'final':final}
 for name,(nb,weight)in graphs.items():scores[name+'_calibrated']=solve(final,ids,nb,weight);print(a.scene,'solved',name,flush=True)
 reference=load(Path('output/graph_transfer_final_20260919')/a.scene/'scores.pt')['scores']['distilled_structure32_calibrated'];err=float((scores['baseline_calibrated']-reference).abs().max());assert err<2e-5,err
 data=dict(texts=bank['texts'],scores=scores,observed=observed,geometry_binding=payload['geometry_binding']);torch.save(data,out/'scores.pt');prep=dict(scene=a.scene,geometry_report_sha256=sha256_file(gp),scores_sha256=sha256_file(out/'scores.pt'),registration=reg,baseline_replay_max_error=err,implementation_sha256=sha256_file(__file__),target_rgb_used=False,geometry_unchanged=True);(out/'preparation.json').write_text(json.dumps(prep,indent=2))
 sys.argv=['diagnose_lerf_native_membership.py','--scene',a.scene,'--output',str(out),'--render-only'];runpy.run_path('scripts/audits/diagnose_lerf_native_membership.py',run_name='__main__')
if __name__=='__main__':
 with torch.no_grad():main()
