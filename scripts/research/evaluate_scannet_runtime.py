"""Score fresh category readouts in two explicitly different label domains."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from plyfile import PlyData
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scannet_constants import compute_split_metrics
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels,volume_weighted_split_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--readout',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);p.add_argument('--point-only',action='store_true');a=p.parse_args();torch.set_num_threads(4)
 d=torch.load(a.readout,map_location='cpu',weights_only=False);spec=json.loads(Path(a.evaluation).read_text());g=d['semantic_geometry'];geo=FrozenGaussianGeometry.load(g['path'],expected_sha256=g['sha256'],device='cpu');observed=d['semantic_observed'].numpy();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 values=dict(source=d['category_logits'].numpy(),map=d['native_logits'].numpy()) if d.get('source_evidence_used','category_logits' in d) else dict(map=d['scores'].numpy());predictions={}
 for arm,score in values.items():
  predictions[arm]={}
  for split,ids in spec['splits'].items():
   cols=[next(i for i,q in enumerate(d['queries']) if q['class_id']==c) for c in ids];pred=np.asarray(ids)[score[:,cols].argmax(1)];pred[~observed]=0;predictions[arm][split]=pred
  np.savez_compressed(out/(arm+'_gaussian_predictions.npz'),**predictions[arm])
 # Coordinates are needed to bind predictions. Labels are consumed only below,
 # after the map's element predictions have been materialized.
 vertices=PlyData.read(spec['ground_truth'])['vertex'].data;xyz=np.stack([vertices[k] for k in ['x','y','z']],1);distance,index=cKDTree(geo.xyz.numpy()).query(xyz,workers=4)
 points={arm:{s:pred[index] for s,pred in split.items()} for arm,split in predictions.items()}
 for arm,pred in points.items():np.savez_compressed(out/(arm+'_point_predictions.npz'),**pred)
 truth=np.asarray(vertices['label']);metrics={arm:{s:compute_split_metrics(pred,truth,spec['splits'][s]) for s,pred in splits.items()} for arm,splits in points.items()}
 volume=None
 if not a.point_only:
  pseudo,_=assign_vala_pseudo_labels(geo.xyz.numpy(),geo.scales.numpy(),geo.quaternions.numpy(),xyz,truth,radius_factor=5.,candidate_k=1000,fallback_k=1,class_balance=True,chunk_size=512)
  weights=(geo.scales.prod(1)*geo.opacities.flatten()).numpy();volume={arm:{s:volume_weighted_split_metrics(pseudo,pred,weights,spec['splits'][s]) for s,pred in splits.items()} for arm,splits in predictions.items()}
 report=dict(scene=spec['scene'],point_metrics=metrics,gaussian_volume_pseudolabel_metrics=volume,point_count=len(xyz),point_mapping='unrestricted nearest centre; unobserved predicts void and is not excluded',point_distance_quantiles=np.quantile(distance,[.5,.9,.99]).tolist(),point_aggregation='equal point weight, absent GT classes ignored, scene macro',official_protocol_equivalence_verified=False,independent_test=False)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(spec['scene'],{arm:{s:100*r['miou'] for s,r in rows.items()} for arm,rows in metrics.items()},flush=True)
if __name__=='__main__':main()
