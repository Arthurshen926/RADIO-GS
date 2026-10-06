"""Post-evaluation precision/recall only; outputs never select seeds or parameters."""
import json
from pathlib import Path
import numpy as np
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
p=Path('output/next106/summary.json');r=json.loads(p.read_text())
for arm,entry in r['controls'].items():
 rows=[]
 for o in entry['onion_objects']:
  gt=load_binary_mask(Path(o['gt_path']),threshold=10);pred=load_binary_mask(Path(o['pred_path']),threshold=10,grayscale=True);inter=int((gt&pred).sum());rows.append(dict(frame=o['frame'],iou=o['iou'],precision=inter/max(1,int(pred.sum())),recall=inter/max(1,int(gt.sum())),gt_pixels=int(gt.sum()),pred_pixels=int(pred.sum())))
 entry['post_evaluation_diagnostic']=dict(views=rows,mean_precision=float(np.mean([x['precision'] for x in rows])),mean_recall=float(np.mean([x['recall'] for x in rows])))
 print(arm,entry['onion_miou'],entry['post_evaluation_diagnostic']['mean_precision'],entry['post_evaluation_diagnostic']['mean_recall'])
p.write_text(json.dumps(r,indent=2)+'\n')
