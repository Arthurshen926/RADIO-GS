"""Opt-in deployed structure-only soft graph readout; no teacher or GT input."""
import argparse,json
from pathlib import Path
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.querying.query_conditioned_diffusion import QueryConditionedDiffusionConfig,solve_continuous_query_support,knn_feature_distances
from radio_gs.utils.immutable_artifacts import sha256_file

def refine_structure_scores(xyz,structure,observed,scores):
 ids=torch.where(observed.cpu())[0];dist,nn=cKDTree(xyz.detach().cpu().numpy()[ids]).query(xyz.detach().cpu().numpy()[ids],k=17,workers=4);neighbors=torch.from_numpy(nn[:,1:].copy()).long().to(xyz.device);spatial=torch.from_numpy(dist[:,1:].copy()).float().to(xyz.device);spatial_scale=spatial[:,-1].median().clamp_min(1e-8)
 feat=F.normalize(structure[ids].float().to(xyz.device),dim=1);dd=knn_feature_distances(feat,neighbors,distance_chunk_size=1024).square();scale=dd[dd>0].median().clamp_min(1e-8);aff=torch.exp(-dd/scale)*torch.exp(-spatial.square()/(2*spatial_scale.square()));confidence=torch.full((len(ids),),.5,device=xyz.device);result=torch.zeros_like(scores,device='cpu')
 cfg=QueryConditionedDiffusionConfig(kernel='continuous_convex_v2',edge_binarize_threshold=None,laplacian_weight=1.,cg_iterations=64,hard_observation_confidence=.99)
 for qi in range(scores.shape[1]):
  unary=scores[ids,qi].to(xyz.device);result[ids,qi]=solve_continuous_query_support(unary,confidence,neighbors,aff,unary,config=cfg).cpu()
 return result,dict(observed_nodes=len(ids),edges=neighbors.numel(),optional_graph_cache_bytes=neighbors.numel()*(8+4),cache_format='int64 neighbor IDs plus float32 affinity; reconstructible from field and geometry',teacher_inputs=False,feature_scale=float(scale))
def main():
 p=argparse.ArgumentParser();p.add_argument('--field',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path);a=p.parse_args();torch.set_num_threads(4)
 assert not a.output.exists();field=torch.load(a.field,map_location='cpu',weights_only=False);data=torch.load(a.scores,map_location='cpu',weights_only=False);geo=FrozenGaussianGeometry.load(a.geometry,expected_sha256=sha256_file(a.geometry),device='cuda:0');geo.assert_binding(field['geometry_binding']);geo.assert_binding(data['geometry_binding']);sd=field['state_dict'];assert sd['latent'].shape[1]==96
 result,report=refine_structure_scores(geo.xyz,sd['latent'][:,64:],sd['observed'],data['scores']['final']);a.output.mkdir(parents=True);torch.save(dict(scores=result,texts=data['texts'],geometry_binding=field['geometry_binding']),a.output/'scores.pt')
 if a.reference:
  target=torch.load(a.reference,map_location='cpu',weights_only=False)['scores']['distilled_structure32_calibrated'];report['replay_max_abs_error']=float((target-result).abs().max());assert report['replay_max_abs_error']<2e-5
 report.update(field_sha256=sha256_file(a.field),input_scores_sha256=sha256_file(a.scores),implementation_sha256=sha256_file(__file__),scores_sha256=sha256_file(a.output/'scores.pt'),opt_in_only=True)
 (a.output/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':
 with torch.no_grad():main()
