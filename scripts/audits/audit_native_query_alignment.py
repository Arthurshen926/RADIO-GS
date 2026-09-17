"""Separate source coverage from native-to-learned query-space distortion."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--run',default='all_four_shared');p.add_argument('--device',default='cuda:3');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
root=a.root;device=a.device;torch.set_num_threads(4);receipt=json.loads((root/'episodes.json').read_text())
if sha256_file(root/'episodes.pt')!=receipt['sha256']:raise ValueError('episode identity')
data=torch.load(root/'episodes.pt',weights_only=False);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());run=json.loads((root/a.run/'report.json').read_text())
geometry=FrozenGaussianGeometry.load(a.geometry,expected_sha256=auth['metadata']['frozen_geometry_sha256'],device=device)
field,_=load_element_retrieval_field(root/a.run/'field.pt',expected_sha256=run['field_sha256'],geometry=geometry,expected_query_encoder=json.loads((root/'native_query_encoder.json').read_text()),device=device)
train_views={info['frames'].index(f) for f in info['train_frames']};regions=[i for i,v in enumerate(data['proposal_views']) if int(v) in train_views]
native=F.normalize(data['queries']['image'].float().to(device),dim=-1);hard=data['region_probability'].to(device)>.7;seen=data['seen'].to(device);rows=[]
with torch.no_grad():
 adapted=F.normalize(field.query_adapters['image'](native),dim=-1)
 for e in data['episodes']:
  if e['split']!='validation' or e['modality']!='image':continue
  q=e['query_region'];target=e['target_region'];target_view=e['target_view'];ious=[];intersections=[]
  for region in regions:
   shared=seen[int(data['proposal_views'][region])]&seen[target_view]
   left=hard[region]&shared;right=hard[target]&shared;intersection=(left&right).sum();union=(left|right).sum()
   ious.append(float(intersection/union.clamp_min(1)));intersections.append(int(intersection))
  compatible=torch.tensor([iou>=.3 and count>=20 for iou,count in zip(ious,intersections)],device=device)
  native_scores=native[regions]@native[q];adapted_scores=adapted[regions]@adapted[q]
  result={'query_region':q,'target_region':target,'max_train_geometry_iou':max(ious),'compatible_training_regions':int(compatible.sum())}
  for name,scores in [('native',native_scores),('adapted',adapted_scores)]:
   order=scores.argsort(descending=True);rank=torch.where(compatible[order])[0]
   result[name]={'best_region':regions[int(order[0])],'best_geometry_iou':ious[int(order[0])],
                 'first_compatible_rank':int(rank[0])+1 if len(rank) else None,'top_cosine':float(scores[order[0]])}
  rows.append(result)
result={'schema':'native_query_alignment_diagnostic.v1','episodes':rows,'field_sha256':run['field_sha256'],'episode_sha256':receipt['sha256'],'training_updates':False,'target_geometry_used_only_for_diagnostic_correspondence':True,'ground_truth_identity':False}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))
