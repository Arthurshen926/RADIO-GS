"""Display source-only native masks at conflicting shared COLMAP observations."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916/ramen')
auditpath=Path('paper/artifacts/ramen_source_track_mask_consistency_20260916.json');audit=json.loads(auditpath.read_text());path=root/'caption_grounded_regions.pt';assert sha256_file(path)==audit['input_bindings'][str(path)];cap=torch.load(path,weights_only=False)
source=json.loads((root/'source_frames.json').read_text());frames={r['frame_idx']:r for r in source['frames']}
examples=[next(e for e in audit['examples'] if e['phrase']==phrase) for phrase in ['bowl of noodles','plates of food']]
out=Path('/mnt/pool/sqy/results/RADIO-GS/output/source_track_conflicts_20260916.png');assert not out.exists()
fig,axes=plt.subplots(len(examples),2,figsize=(12,9));bindings={}
for row,ex in enumerate(examples):
 qi=cap['phrases'].index(ex['phrase'])
 for col,kind in enumerate(['positive','negative']):
  obs=ex[kind][0];frame=obs['frame'];src=frames[frame];imagepath=Path(source['image_dir'])/src['source_file'];assert sha256_file(imagepath)==src['source_sha256'];bindings[str(imagepath)]=src['source_sha256'];im=Image.open(imagepath).convert('RGB');mask=np.zeros((im.height,im.width),dtype=bool)
  for i,r in enumerate(cap['records']):
   if r['frame']==frame and qi in r['phrase_indices'] and r['confidence']>=.7 and cap['masks'][i].sum()>=20:mask|=np.unpackbits(r['packed_native'].numpy(),axis=-1)[:,:im.width].astype(bool)
  ax=axes[row,col];ax.imshow(im);overlay=np.zeros((*mask.shape,4));overlay[mask]=[1,0,0,.35];ax.imshow(overlay);ax.plot(*obs['xy'],marker='+',color='cyan',markersize=18,markeredgewidth=2)
  ax.set_title(f"{ex['phrase']} | point {ex['point_id']}\nframe {frame}: {kind} source mask evidence");ax.axis('off')
fig.tight_layout();fig.savefig(out,dpi=110);out.with_suffix('.json').write_text(json.dumps(dict(schema='source_track_conflict_visualization.v1',image_sha256=sha256_file(out),source_images=bindings,examples=examples,audit_sha256=sha256_file(auditpath),implementation_sha256=sha256_file(__file__)),indent=2)+'\n')
print(out)
