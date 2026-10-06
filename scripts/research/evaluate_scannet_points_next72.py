"""Frozen category readouts transferred to ScanNet label vertices by nearest centre.

Point-weighted diagnostic adapter: no Gaussian-volume weighting or GT-to-Gaussian
pseudo-labels. Matching a named external protocol still requires its scene/domain
and aggregation comparison; this script does not label itself an official score.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
from plyfile import PlyData
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scannet_constants import NYU40_ID_TO_NAME,OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS,compute_split_metrics

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',default='output/typed_evidence_next44/soft_identity');p.add_argument('--output',default='output/scannet_points_next72');p.add_argument('--parent',action='store_true');p.add_argument('--transfer',choices=['nearest','lesv64'],default='nearest');a=p.parse_args();torch.set_num_threads(4);results={}
 for folder in sorted(Path(a.base).glob('scannet_*')):
  scene=folder.name.removeprefix('scannet_');data=load(folder/'field.pt');gr=json.loads(Path(data['geometry_report']).read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cpu');score_path=folder/'anchored/pipeline_calibrated/scores.pt' if a.parent else folder/'elements.pt';readout=load(score_path);values=readout['scores']['hierarchical_pipeline'] if a.parent else readout['scores'];texts=readout['texts']
  if 'semantic_geometry' in readout:
   b=readout['semantic_geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cpu')
  observed=readout.get('semantic_observed',load(data['base_field'])['state_dict']['observed']).bool().cpu().numpy()
  # Read public point coordinates before accessing their labels. No distances are
  # tuned to label quality; the unrestricted nearest-centre rule is explicit.
  gtpath=Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/(scene+'_vh_clean_2.labels.ply');vertices=PlyData.read(str(gtpath))['vertex'].data;xyz=np.stack([vertices[k] for k in ['x','y','z']],1);dist,idx=cKDTree(geo.xyz.numpy()).query(xyz,workers=4)
  out=Path(a.output)/scene;out.mkdir(parents=True,exist_ok=True);pred={}
  for split,ids in SPLITS.items():
   ids=list(ids);cols=[texts.index(NYU40_ID_TO_NAME[c]) for c in ids];per_element=np.asarray(ids)[values[:,cols].argmax(1).numpy()];per_element[~observed]=0;pred[split]=per_element[idx]
  if a.transfer=='lesv64':
   tree=cKDTree(geo.xyz.numpy());opacity=geo.opacities.numpy().reshape(-1);prob={split:values[:,[texts.index(NYU40_ID_TO_NAME[c]) for c in ids]].softmax(1).numpy()*observed[:,None] for split,ids in SPLITS.items()}
   for begin in range(0,len(xyz),4096):
    distance,index=tree.query(xyz[begin:begin+4096],k=min(64,len(geo.xyz)),workers=4);weight=np.exp(-.5*distance**2)*opacity[index];known_mass=(weight*observed[index]).sum(1)
    for split,ids in SPLITS.items():
     score=(prob[split][index]*weight[:,:,None]).sum(1);label=np.asarray(ids)[score.argmax(1)];label[known_mass<=1e-8]=0;pred[split][begin:begin+4096]=label
  np.savez_compressed(out/'predictions.npz',**pred);truth=np.asarray(vertices['label']);metrics={s:compute_split_metrics(v,truth,SPLITS[s]) for s,v in pred.items()}
  row=dict(scene=scene,metrics=metrics,point_count=len(xyz),observed_nearest_fraction=float(observed[idx].mean()),nearest_distance_quantiles=dict(zip(['p50','p90','p99'],np.quantile(dist,[.5,.9,.99]).tolist())),unobserved_prediction=0,unobserved_points_excluded=False,transfer=('nearest Gaussian centre, no distance cutoff' if a.transfer=='nearest' else 'LESV-described K64 Euclidean neighbours, exp(-distance^2/2)*opacity, softmax class logits; unobserved candidates contribute no class probability'),score_source=str(score_path),aggregation='equal point weight within scene, absent GT classes ignored',official_protocol_equivalence_verified=False)
  (out/'report.json').write_text(json.dumps(row,indent=2)+'\n');results[scene]=row;print(scene,{s:100*m['miou'] for s,m in metrics.items()},flush=True)
 summary=dict(scenes=results,scene_macro={s:float(np.mean([r['metrics'][s]['miou'] for r in results.values()])) for s in SPLITS},official_protocol_equivalence_verified=False);(Path(a.output)/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
