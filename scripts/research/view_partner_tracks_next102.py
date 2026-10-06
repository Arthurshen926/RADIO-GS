"""Source-only largest tracker disagreements; no evaluation image selection."""
import json
from pathlib import Path
import torch,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
fig,axs=plt.subplots(2,3,figsize=(15,8));report=[]
for row,(scene,qi) in enumerate([('ramen',11),('figurines',20)]):
 key='lerf_'+scene;a=load(Path('output/uncertain_tracks_next99/all')/key/f'{qi}.pt');b=load(Path('output/partner_tracks_next102')/key/f'{qi}.pt');paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/scene/'all.json').read_text())['images']};values=[]
 for f,r in a['frames'].items():
  t=b['frames'].get(f)
  if not r['present'] or not t or not t['present']:continue
  am=r['probability']>.5;bm=t['probability']>.5
  values.append((float((am&bm).sum()/(am|bm).sum().clamp_min(1)),f))
 chosen=[]
 for score,f in sorted(values):
  if any(abs(f-p)<10 for p in chosen):continue
  chosen.append(f);ax=axs[row,len(chosen)-1];im=Image.open(paths[f]).convert('RGB');im.thumbnail((700,500));ax.imshow(im)
  for track,color in [(a,'red'),(b,'cyan')]:
   mask=np.array(Image.fromarray((track['frames'][f]['probability']>.5).numpy()).resize(im.size,Image.Resampling.NEAREST));ax.contour(mask,levels=[.5],colors=[color],linewidths=1)
  ax.set_title(f'{a["query"]} source {f}: IoU {score:.2f}');ax.axis('off');report.append(dict(scene=scene,query=a['query'],frame=f,track_iou=score))
  if len(chosen)==3:break
fig.suptitle('Source-only tracker disagreement: red original / cyan partner anchor');fig.tight_layout();out=Path('output/partner_exclusion_next102');out.mkdir(exist_ok=True,parents=True);fig.savefig(out/'source_disagreement.jpg',dpi=140);(out/'source_disagreement.json').write_text(json.dumps(report,indent=2)+'\n')
