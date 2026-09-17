import json,torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('/mnt/pool/sqy/results/RADIO-GS/output/component_retrieval_pipeline_20260915/figurines');d=torch.load(r/'grounded_text_independent_pixels.pt',weights_only=False);g=torch.load(d['grounding_path'],weights_only=False);s=json.loads((r/'source_frames.json').read_text());info=json.loads((r/'teachers.json').read_text());h,w=g['probability'].shape[-2:];fig,axes=plt.subplots(2,2,figsize=(12,9))
for ax,e in zip(axes.flat,[e for e in d['tasks'] if e['text_index']==38]):
 frame=info['frames'][e['target_view']];image=np.asarray(Image.open(Path(s['image_dir'])/f'frame_{frame:05d}.jpg').resize((w*4,h*4)));i=g['frames'].index(frame);raw=(g['probability'][i,38]>.8).numpy();new=d['positive_pixels'][e['grounded_text_task']].reshape(h,w).numpy();ax.imshow(image,extent=(0,w,h,0));ax.contour(raw,levels=[.5],colors=['yellow'],linewidths=1);ax.contour(new,levels=[.5],colors=['lime'],linewidths=1.6);ax.set_title(f'Source frame {frame}: hat, {new.sum()} confirmed pixels');ax.axis('off')
fig.suptitle('Yellow: source SAM3 evidence; green: independently confirmed 2D supervision\nTeacher evidence, not ground truth; original compiler admitted no hat tasks');fig.tight_layout();fig.savefig('paper/artifacts/figurines_independent_hat_support_20260916.png',dpi=150)
