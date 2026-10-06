"""A tracker omitting visible reference-positive anchors has incomplete extent.

Keep positive evidence; scale its exclusion evidence by transported-positive
coverage. This is a source-only reliability estimate, not verified identity.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();plan=json.loads((Path('output/reference_cover_next74')/a.scene/'anchors.json').read_text());w,h=plan['raster_wh'];tracks=torch.load(Path('output/typed_evidence_next44/full_source_tracks')/a.scene/'tracks.pt',map_location='cpu',weights_only=False);rows=[]
for row in plan['views']:
 f=row['frame']
 if f not in tracks['frames'] or not row['candidate_points']:continue
 record=tracks['frames'][f];xy=torch.tensor(row['candidate_points']);weights=torch.tensor(row['candidate_weights']);value=F.interpolate(record['logits'].float()[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0];coverage=float(((value[xy[:,1],xy[:,0]]>0)*weights).sum()/weights.sum());record['negative_reliability']=coverage;rows.append(dict(frame=f,visible_positive_coverage=coverage))
out=Path('output/reference_exclusion_next74')/a.scene;out.mkdir(parents=True,exist_ok=True);torch.save(tracks,out/'tracks.pt');(out/'conditioning.json').write_text(json.dumps(dict(rows=rows,positive_evidence_unchanged=True,mask_logits_unchanged=True,target_rgb_used=False,rule='negative reliability equals source-mask recall of visible dominant-owner reference anchors'),indent=2))
