"""Supplemental boundary, precision and recall diagnostics; no fitting."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion,binary_dilation
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from radio_gs.utils.immutable_artifacts import sha256_file
cross=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=bool)
y,x=np.ogrid[-2:3,-2:3];disk=x*x+y*y<=4
for scene in ['figurines','ramen']:
 root=Path('output/lerf_selected_ladder_20260919')/scene;report=json.loads((root/'report.json').read_text());out=root/'mask_error_diagnostics.json';assert not out.exists();results={}
 for name,res in report['results'].items():
  rows=[]
  for obj in res['objects']:
   gt=load_binary_mask(Path(obj['gt_path']));pred=load_binary_mask(Path(obj['pred_path']),grayscale=True);assert gt.shape==pred.shape and gt.ndim==2
   inter=int((gt&pred).sum());precision=inter/max(int(pred.sum()),1);recall=inter/max(int(gt.sum()),1)
   gb=gt&~binary_erosion(gt,structure=cross,border_value=0);pb=pred&~binary_erosion(pred,structure=cross,border_value=0)
   bp=float((pb&binary_dilation(gb,structure=disk)).sum()/max(int(pb.sum()),1));br=float((gb&binary_dilation(pb,structure=disk)).sum()/max(int(gb.sum()),1));bf=2*bp*br/max(bp+br,1e-12)
   rows.append(dict(query=obj['query'],frame=obj['frame'],precision=precision,recall=recall,boundary_f1=bf,iou=obj['iou']))
  results[name]=dict(macro={k:float(np.mean([r[k] for r in rows])) for k in ['precision','recall','boundary_f1','iou']},observations=rows)
 result=dict(diagnostic_only=True,boundary_rule='4-connected inner mask edge; Euclidean <=2 native-grid pixels correspondence; empty prediction=0',report_sha256=sha256_file(root/'report.json'),implementation_sha256=sha256_file(__file__),results=results);out.write_text(json.dumps(result,indent=2)+'\n');print(scene,{k:v['macro'] for k,v in results.items()},flush=True)
