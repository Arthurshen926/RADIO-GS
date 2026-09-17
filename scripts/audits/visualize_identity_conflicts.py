"""Show every region of the first inconsistent source correspondence component."""
import argparse,json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
audit=json.loads(a.audit.read_text());root=a.root
if sha256_file(root/'episodes.pt')!=audit['episode_sha256']:raise ValueError('episode identity')
data=torch.load(root/'episodes.pt',weights_only=False);source=json.loads((root/'source_frames.json').read_text());info=json.loads((root/'teachers.json').read_text())
if sha256_file(root/'source_frames.json')!=info['source_manifest_sha256']:raise ValueError('source identity')
component=next(c for c in audit['graph']['train']['components'] if not c['track_consistent']);regions=component['regions'];frames=info['frames'];sources={r['frame_idx']:r for r in source['frames']}
scores=torch.nn.functional.normalize(data['queries']['image'].float(),dim=-1)@torch.nn.functional.normalize(data['queries']['text'].float(),dim=-1).T
fig,axes=plt.subplots((len(regions)+3)//4,4,figsize=(16,7));axes=axes.flatten()
for ax,region in zip(axes,regions):
 vi=int(data['proposal_views'][region]);frame=frames[vi];record=sources[frame];path=Path(source['image_dir'])/record['source_file']
 if sha256_file(path)!=record['source_sha256']:raise ValueError('RGB identity')
 mask=data['masks'][region].numpy();h,w=mask.shape;rgb=np.array(Image.open(path).resize((w,h)))/255
 ax.imshow(rgb*.3+rgb*.7*mask[:,:,None]);ax.contour(mask,levels=[.5],colors=['lime'],linewidths=1)
 word=data['queries']['queries'][int(scores[region].argmax())]
 ax.set_title(f'region {region}, view {frame}\n{word}');ax.axis('off')
for ax in axes[len(regions):]:ax.axis('off')
fig.suptitle('Pairwise matching component with multiple regions from the same view\nCannot be treated as one physical instance without further evidence. Text titles are teacher predictions.')
fig.tight_layout();fig.savefig(a.output,dpi=150)
