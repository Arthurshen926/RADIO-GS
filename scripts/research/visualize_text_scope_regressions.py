"""Inspect legal source predictions; no target labels or target photographs."""
import json,torch,numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('output/text_scope_regressions');root.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(6,2,figsize=(14,18));row=0;records=[]
for scene,text in [('figurines','rubber duck with buoy'),('teatime','coffee')]:
    c=json.loads(Path(f'configs/consolidated/{scene}.scene.json').read_text());images=json.loads(Path(c['source_rgb_authority']).read_text())['images'];images={int(r['image_id'].split('_')[-1]):r['path'] for r in images};evidence=Path(f'output/conditioned_identity/full_source/{scene}/identity_evidence');candidates=[]
    for f in c['source_frames']:
        d=torch.load(evidence/f'identity_{f}.pt',map_location='cpu',weights_only=False);q=d['queries'].index(text);r=d['instances'][q]
        if len(r['scores']):candidates.append((f,float(r['scores'].max()),r['masks'].float().amax(0),r.get('unknown_masks')))
    # Three temporal quantiles among views that contain an admitted instance.
    for j in [0,len(candidates)//2,len(candidates)-1]:
        frame,confidence,mask,unknown=candidates[j];rgb=Image.open(images[frame]).convert('RGB').resize((512,384));m=np.asarray(Image.fromarray(mask.numpy()).resize((512,384),Image.Resampling.BILINEAR));axes[row,0].imshow(rgb);axes[row,1].imshow(rgb);axes[row,1].contour(m,levels=[.5],colors=['red'],linewidths=1);axes[row,0].set_title(f'{scene}: {text}, source {frame}');axes[row,1].set_title(f'admitted mask, detector confidence {confidence:.3f}');axes[row,0].axis('off');axes[row,1].axis('off');records.append(dict(scene=scene,query=text,frame=frame,confidence=confidence,rgb=images[frame]));row+=1
fig.tight_layout();fig.savefig(root/'source_masks.jpg',dpi=105);(root/'gallery.json').write_text(json.dumps(dict(records=records,selection='first/middle/last admitted source view',target_labels_used=False),indent=2)+'\n')
