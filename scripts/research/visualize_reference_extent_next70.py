"""Post-evaluation diagnostic only: never feeds source training or selection."""
from pathlib import Path
import json
import numpy as np,torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
rows=[]
for scene,query in [('teatime','bear nose'),('teatime','coffee'),('figurines','miffy')]:
 base=Path('output/dual_observation_next68/seeded')/('lerf_'+scene);d=torch.load(base/'field.pt',map_location='cpu',weights_only=False);src=json.loads((Path(d['root'])/'source_frames.json').read_text())
 records=json.loads((base/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']['selected']['objects'];r=next(v for v in records if v['query']==query)
 gt=load_binary_mask(Path(r['gt_path']),threshold=10);pred=load_binary_mask(Path(r['pred_path']),threshold=10,grayscale=True);H,W=gt.shape
 path=Path(src['image_dir'])/(r['frame']+'.jpg');rgb=np.array(Image.open(path).convert('RGB').resize((W,H)))
 yy,xx=np.where(gt|pred);x0,x1=max(0,xx.min()-30),min(W,xx.max()+31);y0,y1=max(0,yy.min()-30),min(H,yy.max()+31)
 rows.append((scene,query,r['frame'],rgb[y0:y1,x0:x1],gt[y0:y1,x0:x1],pred[y0:y1,x0:x1]))
fig,axes=plt.subplots(3,3,figsize=(12,10))
for i,(scene,q,frame,rgb,gt,pred) in enumerate(rows):
 for j in range(3):
  axes[i,j].imshow(rgb)
  if j:
   mask=gt if j==1 else pred;rgba=np.zeros((*mask.shape,4));rgba[mask]=[1,0,0,.4];axes[i,j].imshow(rgba)
  axes[i,j].set_title([q+' / '+frame,'Annotated extent','Retained selected projection'][j]);axes[i,j].axis('off')
fig.tight_layout();fig.savefig('output/zoom_grounding_next70/reference_extent_diagnostic.png',dpi=120);plt.close(fig)
