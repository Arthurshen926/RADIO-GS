"""Source candidate sheets for assistant review, never target benchmark masks."""
import json
from pathlib import Path
import torch,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/pixel_evidence_rebuild_20260917/figurines/field');ep=root/'caption_region_evidence.pt';e=torch.load(ep,weights_only=False);source=json.loads((root/'source_frames.json').read_text());srcs={r['frame_idx']:r for r in source['frames']};out=Path('output/expert_plan_20260918/review_candidates');out.mkdir(parents=True,exist_ok=False);rows=[]
for frame in [11,21,31,174]:
 ids=[i for i,r in enumerate(e['records']) if r['frame']==frame and any(any(k in e['phrases'][qi].lower() for k in ['bag','packet','puff','candy']) for qi in r['phrase_indices']) and int(e['masks'][i].sum())>=20]
 # Exact coarse duplicates have identical review pixels. Keep names together.
 unique={}
 for i in ids:unique.setdefault(e['masks'][i].numpy().tobytes(),[]).append(i)
 groups=list(unique.values());src=srcs[frame];p=Path(source['image_dir'])/src['source_file'];assert sha256_file(p)==src['source_sha256'];image=Image.open(p).convert('RGB');h,w=e['masks'].shape[-2:];rgb=np.asarray(image.resize((w,h))).astype(float)/255
 for page in range(0,len(groups),9):
  subset=groups[page:page+9];fig,axes=plt.subplots(3,3,figsize=(15,12));
  for ax in axes.flat:ax.axis('off')
  for ax,rr in zip(axes.flat,subset):
   mask=e['masks'][rr[0]].numpy().astype(bool);overlay=rgb.copy();overlay[mask]=.5*rgb[mask]+.5*np.array([1.,.2,.1]);phrases=sorted({e['phrases'][q] for i in rr for q in e['records'][i]['phrase_indices']});ax.imshow(overlay);ax.set_title('IDs '+str(rr)+'\n'+' / '.join(phrases)[:120],fontsize=8);rows.append(dict(frame=frame,region_ids=rr,phrases=phrases))
  fig.suptitle('figurines source frame '+str(frame));fig.tight_layout();fig.savefig(out/f'frame_{frame}_{page//9}.png',dpi=120);plt.close(fig)
(out/'manifest.json').write_text(json.dumps(dict(source_only=True,benchmark_data_used=False,evidence_sha256=sha256_file(ep),candidates=rows),indent=2)+'\n');print([(f,len([r for r in rows if r['frame']==f])) for f in [11,21,31,174]])
