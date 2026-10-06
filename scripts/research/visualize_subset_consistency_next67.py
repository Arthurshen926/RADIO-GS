"""Post-evaluation error breakdown and illustrative masks; never training inputs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
root=Path('output/subset_consistency_next67');base=Path('output/joint_semantic_next66/adaptive_allrgb/decision_continuation')
rows=[];stats={}
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 key='lerf_'+scene;metrics=[]
 for parent in [base,root/'exposed']:
  m=json.loads((parent/key/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']['selected']['objects'];metrics.append(m)
 results=[];examples=[]
 for old,new in zip(*metrics):
  gt=load_binary_mask(Path(new['gt_path']),threshold=10)
  preds=[load_binary_mask(Path(r['pred_path']),threshold=10,grayscale=True) for r in [old,new]]
  row=[]
  for pred in preds:
   tp=(pred&gt).sum();row.append(dict(precision=float(tp/max(pred.sum(),1)),recall=float(tp/max(gt.sum(),1)),fp=int((pred&~gt).sum()),fn=int((~pred&gt).sum())))
  results.append(row);examples.append((new['iou']-old['iou'],old,new,gt,preds))
 stats[scene]={arm:{metric:float(np.mean([r[i][metric] for r in results])) for metric in ['precision','recall','fp','fn']} for i,arm in enumerate(['parent','exposed'])}
 # One largest improvement and one largest regression, explicitly diagnostic.
 for case in [max(examples,key=lambda v:v[0]),min(examples,key=lambda v:v[0])]:rows.append((scene,case))
(root/'error_breakdown.json').write_text(json.dumps(stats,indent=2)+'\n')
fig,axes=plt.subplots(len(rows),3,figsize=(12,len(rows)*2.5),squeeze=False)
for i,(scene,(_,old,new,gt,preds)) in enumerate(rows):
 for j,(mask,title) in enumerate([(gt,'GT'),(preds[0],f"Before {old['iou']*100:.1f}%"),(preds[1],f"Exposed-support {new['iou']*100:.1f}%")]):
  axes[i,j].imshow(mask,cmap='gray',vmin=0,vmax=1);axes[i,j].set_title(title);axes[i,j].axis('off')
 axes[i,0].text(-.03,.5,scene+'\n'+new['query'],transform=axes[i,0].transAxes,ha='right',va='center',fontsize=9)
fig.suptitle('Diagnostic examples: largest gain and largest regression per scene; no training use',fontsize=13)
fig.tight_layout(rect=[.04,0,1,.98]);fig.savefig(root/'selected_examples.png',dpi=120);plt.close(fig)
print(json.dumps(stats,indent=2))
