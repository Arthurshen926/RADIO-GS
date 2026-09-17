"""Document what polygon-derived reference queries can and cannot measure."""
import json,argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--labels',type=Path,default=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 d=json.loads(a.evaluation.read_text());annotations,categories,h,w=load_labels(a.labels,a.scene);height=round(h*480/w);rows=[];references={i:r for i,r in enumerate(d['references'])}
 # Count annotation regions and disconnected raster supports, not instances.
 def scope(frame,category):
  objects=[x for x in annotations[frame] if x['category']==category];mask=ground_truth_masks(annotations[frame],[category],height,480,h,w)[category].numpy();components,n=label(mask,np.ones((3,3)));sizes=np.bincount(components.ravel())[1:]
  return {'annotation_regions':len(objects),'components_at_least_8_pixels':int((sizes>=8).sum()),'positive_pixels':int(mask.sum())}
 sample=next(iter(d['runs'].values()))['observations']
 for x in sample:
  ref=references[x['reference']];rs=scope(ref['frame'],x['category']);ts=scope(x['target_frame'],x['category']);rows.append({'category':x['category'],'reference_frame':ref['frame'],'target_frame':x['target_frame'],'reference_support':rs,'target_support':ts,'multiple_disconnected_supports':rs['components_at_least_8_pixels']>1 or ts['components_at_least_8_pixels']>1,'prompt_available':x['prompt_available']})
 metrics={}
 for name,r in d['runs'].items():
  groups={}
  for key,select in [('single_connected_support',lambda x:not x['multiple_disconnected_supports']),('multiple_connected_supports',lambda x:x['multiple_disconnected_supports'])]:
   ids=[i for i,row in enumerate(rows) if select(row)];groups[key]={'tasks':len(ids),'is_verified_instance_evaluation':False,'iou':{m:sum(r['observations'][i][m]['iou'] for i in ids)/len(ids) if ids else None for m in r['macro']}}
  metrics[name]=groups
 raw=[]
 for path in sorted((a.labels/a.scene).glob('frame_*.json')):
  js=json.loads(path.read_text());raw.append({'file':str(path),'sha256':sha256_file(path),'object_fields':sorted({k for o in js['objects'] for k in o}),'category_group_pairs':[[o['category'],o.get('group')] for o in js['objects']]})
 result={'schema':'polygon_reference_query_contract_audit.v1','evaluation_sha256':sha256_file(a.evaluation),'scene':a.scene,'tasks':len(rows),'multiple_connected_support_tasks':sum(x['multiple_disconnected_supports'] for x in rows),'missing_registered_prompt_tasks':sum(not x['prompt_available'] for x in rows),'truth_domain':'2D category polygon union; not independently verified 3D/instance tracks','query_source':'reference GT masked crop or GT-derived positive and negative registered points','target_truth_used_for_encoding':False,'cohort_has_absent_category_targets':False,'single_connected_support_does_not_prove_identity':True,'source_annotations':raw,'diagnostic_strata':metrics,'observations':rows}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['source_annotations','diagnostic_strata','observations']})
if __name__=='__main__':main()
