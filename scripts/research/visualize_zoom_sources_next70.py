"""Source-only qualitative examples, selected by teacher confidence, without GT."""
import json
from pathlib import Path
import torch,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import functional as F

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
root=Path('output/zoom_grounding_next70');rows=[];missing=[]
for scene,query in [('figurines','miffy'),('figurines','pirate hat'),('ramen','corn'),('teatime','bear nose'),('teatime','coffee'),('teatime','dall-e brand')]:
 key='lerf_'+scene;data=load(Path('output/taskwise_scope_composition_next38')/key/'field.pt');r=Path(data['root']);src=json.loads((r/'source_frames.json').read_text());paths={f['frame_idx']:Path(src['image_dir'])/f['source_file'] for f in src['frames']}
 choices=[]
 for p in (root/'evidence'/key).glob('identity_*.pt'):
  d=load(p);qi=d['queries'].index(query);v=d['instances'][qi]
  if len(v['scores']):choices.append((float(v['scores'].max()),d,qi))
 if not choices:missing.append((scene,query));continue
 _,d,qi=max(choices,key=lambda v:v[0]);frame=d['frame'];image=np.array(Image.open(paths[frame]).convert('RGB'));H,W=image.shape[:2]
 old=load(Path('output/typed_evidence_next44')/key/f'identity_{frame}.pt')['instances'][qi];new=d['instances'][qi];box=d['windows'][query];x0,y0,x1,y1=box
 masks=[]
 for v in [old,new]:
  mask=v['masks'].float().amax(0) if len(v['scores']) else torch.zeros(new['masks'].shape[-2:]);masks.append(F.interpolate(mask[None,None],size=(H,W),mode='nearest')[0,0].numpy()[y0:y1,x0:x1]>.5)
 rows.append((scene,query,frame,image[y0:y1,x0:x1],masks))
fig,axes=plt.subplots(len(rows),3,figsize=(10,3*len(rows)),squeeze=False)
for i,(scene,query,frame,rgb,masks) in enumerate(rows):
 for j in range(3):
  axes[i,j].imshow(rgb)
  if j:
   mask=masks[j-1];overlay=np.zeros((*mask.shape,4));overlay[mask]=[1,0,0,.4];axes[i,j].imshow(overlay)
  axes[i,j].axis('off');axes[i,j].set_title(['Source crop','Original detector','Crop detector + identity'][j])
 axes[i,0].set_ylabel(f'{scene}/{frame}\n{query}')
 axes[i,0].text(-.03,.5,query,transform=axes[i,0].transAxes,va='center',ha='right')
fig.tight_layout();fig.savefig(root/'source_examples.png',dpi=120);plt.close(fig)
(root/'source_examples.json').write_text(json.dumps(dict(examples=[dict(scene=s,query=q,frame=f) for s,q,f,_,_ in rows],no_qualified_crop_detection=missing,selection='highest qualified crop detector confidence; source-only; no GT'),indent=2)+'\n')
print('missing',missing)
