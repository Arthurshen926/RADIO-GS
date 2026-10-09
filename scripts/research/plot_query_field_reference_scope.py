"""Full-cohort selected-set error maps after sealed scoring, without RGB inputs."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

root=Path('output/query_field_20261009/reference_scope_v1')
assert (root/'scoring_complete.json').exists()
d=json.loads((root/'design.json').read_text());r=json.loads((root/'summary.json').read_text())
arms=['raw_signed','raw_positive','mask_rays','mask_composition']
labels=['Raw signed','Raw positive only','SAM query mask: rays','SAM query mask: composition']
fig,axes=plt.subplots(8,4,figsize=(16,23),layout='constrained')
for row,j in enumerate(d['jobs']):
 s=j['name'];cfg=json.loads(Path(j['configuration']['path']).read_text());scoring=json.loads(Path(j['scoring']['path']).read_text())
 raw=np.asarray(Image.open(scoring['gt_mask']));target=np.any(raw>127,axis=2) if raw.ndim==3 else raw>127
 for col,a in enumerate(arms):
  payload=torch.load(root/'predictions'/s/(s+'__'+a)/'predictions.pt',map_location='cpu',weights_only=True)
  predicted=cv2.resize(payload['response'].numpy(),(target.shape[1],target.shape[0]),interpolation=cv2.INTER_LINEAR)>cfg['alpha_threshold']
  rgb=np.full((*target.shape,3),255,np.uint8)
  rgb[target&predicted]=[46,125,50];rgb[~target&predicted]=[231,111,81];rgb[target&~predicted]=[69,123,157]
  axes[row,col].imshow(rgb);axes[row,col].set_title(f"{s} | IoU {100*r['per_case'][a][s]['selection_iou']:.2f}%",fontsize=10);axes[row,col].axis('off')
  if row==0:axes[row,col].set_title(labels[col]+'\n'+axes[row,col].get_title(),fontsize=11)
fig.suptitle('All eight NVOS requests: actual selected-set projection errors\nSame frozen fields; left: raw strokes, right: richer SAM query input; green TP, orange FP, blue FN',fontsize=14)
out=root/'diagnostics';out.mkdir(exist_ok=True)
for suffix in ['png','pdf']:fig.savefig(out/('all_case_errors.'+suffix),dpi=150)
print(out/'all_case_errors.png')
