import json
from pathlib import Path
import torch,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916/ramen');cap=torch.load(root/'caption_region_evidence.pt',weights_only=False);info=json.loads((root/'teachers.json').read_text());source=json.loads((root/'source_frames.json').read_text());records={r['frame_idx']:r for r in source['frames']}
fig,axes=plt.subplots(3,4,figsize=(16,10))
qi=cap['phrases'].index('white napkin')
for ax,frame in zip(axes.flat,info['train_frames']):
 r=records[frame];p=Path(source['image_dir'])/r['source_file'];assert sha256_file(p)==r['source_sha256'];im=Image.open(p).convert('RGB');vi=info['frames'].index(frame)
 ids=[i for i,r in enumerate(cap['records']) if r['frame']==frame and qi in r['phrase_indices'] and r['confidence']>=.7 and cap['masks'][i].sum()>=20]
 ax.imshow(im)
 if ids:
  mask=cap['masks'][ids].bool().any(0).numpy();mask=np.asarray(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST));overlay=np.zeros((*mask.shape,4));overlay[mask]=[1,0,0,.45];ax.imshow(overlay)
 ax.set_title(f'frame {frame} | white napkin | masks {len(ids)}');ax.axis('off')
fig.tight_layout();fig.savefig('/mnt/pool/sqy/results/RADIO-GS/output/source_white_napkin_atlas_20260916.png',dpi=110)
