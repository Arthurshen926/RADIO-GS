"""Visualize source-only native identity versus full scope takeover."""
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file

torch.set_num_threads(4)
base=Path('output/expert_plan_20260918');report=json.loads((base/'reviewed_bag_identity.json').read_text())
reviewpath=Path('docs/reviews/2026-09-18-bag-scope-track.json');assert sha256_file(reviewpath)==report['review_sha256'];review=json.loads(reviewpath.read_text())
root=Path('output/pixel_evidence_rebuild_20260917/figurines/field');ep=root/'caption_region_evidence.pt';fp=root/'counterfactual_extent_benefit_admission/field.pt'
assert sha256_file(ep)==report['evidence_sha256'] and sha256_file(fp)==report['field_sha256']
e=torch.load(ep,weights_only=False);payload=torch.load(fp,weights_only=False);field=ElementRetrievalField(**payload['architecture']);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());source=json.loads((root/'source_frames.json').read_text());srcs={r['frame_idx']:r for r in source['frames']};h,w=e['masks'].shape[-2:]
fig,axes=plt.subplots(len(review['observations']),3,figsize=(12,9))
with torch.no_grad():
 q=e['text_embeddings'][e['phrases'].index(review['query'])][None].float()
 native=(field.readout_log_scale['text'].clamp(-4,6).exp()*(field.native_cosine(q)-field.native_query_thresholds['text'](q))).sigmoid()[:,0]
 complete=field.retrieve(q,'text')['scores'][:,0]
 for row,obs in enumerate(review['observations']):
  frame=obs['frame'];src=srcs[frame];ip=Path(source['image_dir'])/src['source_file'];assert sha256_file(ip)==src['source_sha256'];rgb=np.asarray(Image.open(ip).convert('RGB').resize((w,h))).astype(float)/255
  rec=auth['views'][info['frames'].index(frame)];hp=root/rec['relative_path'];assert sha256_file(hp)==rec['sha256'];hit=torch.load(hp,weights_only=False);pid,gid,weight=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float();mass=torch.zeros(h*w).scatter_add_(0,pid,weight);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,len(field.latent))).coalesce();support=torch.sparse.mm(mat,field.observed.float()[:,None])[:,0]
  masks=[e['masks'][obs['region']].bool()]
  for value in [native,complete]:
   score=torch.sparse.mm(mat,(value*field.observed)[:,None])[:,0]/support.clamp_min(1e-8);masks.append(((score>.5)&(support>.5)).reshape(h,w))
  for col,(title,mask) in enumerate(zip(['Reviewed teacher scope','Native text','Complete text'],masks)):
   overlay=rgb.copy();mm=mask.numpy();overlay[mm]=.5*overlay[mm]+.5*np.array([1.,.15,.05]);axes[row,col].imshow(overlay);axes[row,col].set_title(f'Frame {frame}: {title}');axes[row,col].axis('off')
fig.suptitle('Query: plastic bag | selected source diagnostic, not benchmark GT');fig.tight_layout();out=base/'reviewed_bag_identity.png';assert not out.exists();fig.savefig(out,dpi=140);plt.close(fig)
(base/'reviewed_bag_identity_visual.json').write_text(json.dumps(dict(figure=str(out),sha256=sha256_file(out),identity_report_sha256=sha256_file(base/'reviewed_bag_identity.json'),implementation_sha256=sha256_file(__file__)),indent=2)+'\n');print(out)
