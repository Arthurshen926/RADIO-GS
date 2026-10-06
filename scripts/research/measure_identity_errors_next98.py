"""Post-evaluation precision/recall decomposition; no prediction selection."""
import json
from pathlib import Path
import numpy as np
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
results=[]
for scene,queries in [('ramen',['kamaboko','sake cup']),('figurines',['waldo']),('waldo_kitchen',['spatula','pour-over vessel'])]:
 for query in queries:
  row=dict(scene=scene,query=query,arms={})
  for name,base in [('parent','output/coverage_scope_next97/high_footprint'),('direct','output/identity_scope_next98/high_footprint'),('description','output/description_identity_scope_next98/high_footprint'),('category','output/category_identity_scope_next98/high_footprint')]:
   data=json.loads((Path(base)/('lerf_'+scene)/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']['selected']['objects'];values=[]
   for r in data:
    if r['query']!=query:continue
    gt=load_binary_mask(Path(r['gt_path']),threshold=10);pred=load_binary_mask(Path(r['pred_path']),threshold=10,grayscale=True);intersection=np.logical_and(gt,pred).sum();values.append(dict(frame=r['frame'],iou=r['iou'],precision=float(intersection/max(1,pred.sum())),recall=float(intersection/max(1,gt.sum())),gt_pixels=int(gt.sum()),pred_pixels=int(pred.sum())))
   row['arms'][name]=dict(objects=values,mean_iou=float(np.mean([v['iou'] for v in values])),mean_precision=float(np.mean([v['precision'] for v in values])),mean_recall=float(np.mean([v['recall'] for v in values])))
  results.append(row)
Path('output/next98/error_decomposition.json').write_text(json.dumps(results,indent=2)+'\n');print([(r['query'],[(a,round(v['mean_precision'],3),round(v['mean_recall'],3)) for a,v in r['arms'].items()]) for r in results])
