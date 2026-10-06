"""Source-only whole/part and multi-instance candidate sheets for visual review."""
import argparse,json,re
from pathlib import Path
import torch,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--frames',required=True);p.add_argument('--keywords',required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/v2_scope_rebuild_20260918')/a.scene/'field';ep=root/'caption_region_evidence.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];e=torch.load(ep,weights_only=False);assert e['source_only'] and not e['benchmark_data_used'];source=json.loads((root/'source_frames.json').read_text());srcs={r['frame_idx']:r for r in source['frames']};train=json.loads((root/'teachers.json').read_text())['train_frames'];frames=list(map(int,a.frames.split(',')));assert set(frames)<=set(train);a.output.mkdir(parents=True,exist_ok=False);records=[]
for frame in frames:
 unique={}
 for i,r in enumerate(e['records']):
  if r['frame']!=frame or int(e['masks'][i].sum())<10:continue
  if not any(any(re.search(r'\b'+re.escape(word),e['phrases'][q].lower()) for word in a.keywords.split(',')) for q in r['phrase_indices']):continue
  unique.setdefault(e['masks'][i].numpy().tobytes(),[]).append(i)
 groups=list(unique.values());src=srcs[frame];ip=Path(source['image_dir'])/src['source_file'];assert sha256_file(ip)==src['source_sha256'];h,w=e['masks'].shape[-2:];rgb=np.asarray(Image.open(ip).convert('RGB').resize((w*3,h*3))).astype(float)/255
 for start in range(0,len(groups),12):
  sheet=a.output/f'frame_{frame}_{start//12}.png';fig,axes=plt.subplots(3,4,figsize=(16,10));
  for ax in axes.flat:ax.axis('off')
  for ax,ids in zip(axes.flat,groups[start:start+12]):
   mm=np.asarray(Image.fromarray(e['masks'][ids[0]].numpy().astype('uint8')).resize((w*3,h*3),Image.Resampling.NEAREST)).astype(bool);overlay=rgb.copy();overlay[mm]=.55*overlay[mm]+.45*np.array([1,.1,.05]);phrases=sorted({e['phrases'][q] for i in ids for q in e['records'][i]['phrase_indices']});ax.imshow(overlay);ax.set_title(f'IDs {ids}\n'+ ' / '.join(phrases)[:120],fontsize=8);records.append(dict(frame=frame,region_ids=ids,phrases=phrases,sheet=str(sheet),source_image=str(ip),source_image_sha256=src['source_sha256']))
  fig.suptitle(f'{a.scene} source frame {frame} | automatic candidates, not verified masks');fig.tight_layout();fig.savefig(sheet,dpi=120);plt.close(fig)
manifest=dict(source_only=True,benchmark_data_used=False,training=False,evidence_path=str(ep),evidence_sha256=sha256_file(ep),implementation_sha256=sha256_file(__file__),candidates=records,sheets={str(p):sha256_file(p) for p in a.output.glob('*.png')});(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print({f:sum(r['frame']==f for r in records) for f in frames})
