"""Reviewable source-only disagreements; never invent manual target/identity labels."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.utils.immutable_artifacts import sha256_file
base=Path('output/current_recipe_audit_20260918');cases=[]
for scene in ['figurines','ramen']:
 root=Path('output/pixel_evidence_rebuild_20260917')/scene/'field';d=json.loads((base/scene/'source_collisions.json').read_text());ep=root/'caption_region_evidence.pt';assert sha256_file(ep)==d['evidence_sha256'];e=torch.load(ep,weights_only=False);source=json.loads((root/'source_frames.json').read_text());sources={r['frame_idx']:r for r in source['frames']};out=base/scene/'casebook';out.mkdir(exist_ok=False)
 selected=[p for p in sorted(d['identical_membership_coobserved_pairs'],key=lambda p:p['minimum_source_mask_iou']) if p['minimum_source_mask_iou']<.5]
 for ci,pair in enumerate(selected):
  comparisons=sorted(pair['coobserved'],key=lambda r:r['iou'])[:3];fig,axes=plt.subplots(len(comparisons),3,figsize=(15,4*len(comparisons)),squeeze=False);images=[]
  for row,c in enumerate(comparisons):
   frame=c['frame'];src=sources[frame];path=Path(source['image_dir'])/src['source_file'];assert sha256_file(path)==src['source_sha256'];rgb=Image.open(path).convert('RGB');axes[row,0].imshow(rgb);axes[row,0].set_title(f'{scene}: source frame {frame}');images.append(dict(path=str(path),sha256=src['source_sha256']))
   for col,side in enumerate(['left','right'],1):
    qi=pair[side]['query'];ids=[i for i,r in enumerate(e['records']) if r['frame']==frame and qi in r['phrase_indices'] and r['confidence']>=.7 and int(e['masks'][i].sum())>=20];mask=e['masks'][ids].bool().any(0).numpy();display=np.asarray(rgb.resize((mask.shape[1],mask.shape[0]))).astype(float)/255;overlay=display.copy();overlay[mask]=.5*display[mask]+.5*np.array([1.,.2,.1]);axes[row,col].imshow(overlay);axes[row,col].set_title(pair[side]['phrase'])
   for ax in axes[row]:ax.axis('off')
  fig.suptitle(f'Identical final element membership; source mask min IoU {pair["minimum_source_mask_iou"]:.3f}\nAutomatic evidence only; physical identity and intended scope unverified');fig.tight_layout();fp=out/f'case_{ci:02d}.png';fig.savefig(fp,dpi=120);plt.close(fig)
  cases.append(dict(scene=scene,case=ci,pair=pair,images=images,figure=str(fp),figure_sha256=sha256_file(fp),semantic_description={'left':pair['left']['phrase'],'right':pair['right']['phrase']},physical_instance_id=None,scope_granularity=None,visibility_judgment=None,manual_review_status='pending',training_use=False))
(base/'scope_casebook.json').write_text(json.dumps(dict(source_only=True,benchmark_data_used=False,human_verified=False,cases=cases,implementation_sha256=sha256_file(__file__)),indent=2)+'\n')
print('casebook',len(cases),'automatic cases, manual review pending')
