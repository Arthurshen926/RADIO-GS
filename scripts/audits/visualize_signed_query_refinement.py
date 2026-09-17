"""Show the first heldout prompt and the unresolved signed-constraint case."""
import argparse,json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.models.signed_query_refinement import refine_signed_query,retrieve_latent_query
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.training.retrieval_supervision import episode_membership_targets
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
root=a.root;device='cuda:0';torch.set_num_threads(4);run='all_four_shared'
evaluation=json.loads(a.evaluation.read_text());records=[r for r in evaluation['runs'][run]['episodes'] if r['modality']=='prompt']
selected=[records[0]]
for predicate in [lambda r:r['signed_evidence']['refinement_applied'],lambda r:r['signed_evidence']['refinement_applied'] and not all(r['signed_evidence']['confidence_targets_satisfied'])]:
 for record in records:
  if predicate(record) and record not in selected:selected.append(record);break
if sha256_file(root/'episodes.pt')!=evaluation['episode_sha256']:raise ValueError('episode identity')
data=torch.load(root/'episodes.pt',weights_only=False);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());source=json.loads((root/'source_frames.json').read_text());sources={r['frame_idx']:r for r in source['frames']}
if sha256_file(root/'responsibility.json')!=info['authority_sha256']:raise ValueError('authority identity')
geometry=FrozenGaussianGeometry.load(a.geometry,expected_sha256=auth['metadata']['frozen_geometry_sha256'],device=device)
field,_=load_element_retrieval_field(root/run/'field.pt',expected_sha256=evaluation['runs'][run]['field_sha256'],geometry=geometry,expected_query_encoder=json.loads((root/'native_query_encoder.json').read_text()),device=device)
h=auth['metadata']['feature_height'];w=auth['metadata']['feature_width'];n=auth['num_gaussians'];matrices=[]
for record in auth['views']:
 path=root/record['relative_path']
 if sha256_file(path)!=record['sha256']:raise ValueError('view identity')
 hit=torch.load(path,map_location=device,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['base_weights'].float();mass=torch.zeros(h*w,device=device).scatter_add_(0,pid,weight)
 matrices.append(torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,n)).coalesce())
def rgb(view):
 frame=auth['frame_indices'][view];record=sources[frame];path=Path(source['image_dir'])/record['source_file']
 if sha256_file(path)!=record['source_sha256']:raise ValueError('RGB identity')
 return np.array(Image.open(path).resize((w,h)))
fig,axes=plt.subplots(len(selected),5,figsize=(16,3.3*len(selected)),squeeze=False)
with torch.no_grad():
 for row,record in enumerate(selected):
  e=data['episodes'][record['episode_index']];q=e['query_region'];qview=int(data['proposal_views'][q]);v=e['target_view'];pixels=data['queries']['seed_pixels'][q].tolist();query=data['queries']['prompt'][q:q+1].float().to(device)
  matrix=matrices[qview];indices=matrix.indices();weight=matrix.values();rows=[];weights=[]
  for pixel in pixels:
   chosen=indices[0]==pixel;rows.append(indices[1,chosen]);weights.append(weight[chosen])
  result=refine_signed_query(field,query,rows,weights,[1.,0.]);before=field.retrieve(query,'prompt')['scores'][:,0];after=retrieve_latent_query(field,result['latent_query'])['scores'][:,0]
  support=torch.sparse.mm(matrices[v],field.observed.float()[:,None])[:,0]
  target,known,_,_=episode_membership_targets(e,data['masks'].flatten(1).to(device),data['region_probability'].to(device),field.observed,support)
  labels=np.full((h*w,3),.45);labels[(known&target.bool()).cpu()]=[.15,.9,.25];labels[(known&~target.bool()).cpu()]=[.9,.2,.15]
  axes[row,0].imshow(rgb(qview))
  for pixel,color,marker in zip(pixels,['lime','red'],['+','x']):axes[row,0].scatter(pixel%w,pixel//w,c=color,marker=marker,s=120,linewidths=3)
  axes[row,0].set_title('Authorized source +/- points')
  axes[row,1].imshow(rgb(v));axes[row,1].set_title('Heldout target RGB')
  axes[row,2].imshow(labels.reshape(h,w,3));axes[row,2].set_title('Pseudo +/- labels; gray unknown')
  for ci,posterior in enumerate([before,after]):
   image=torch.sparse.mm(matrices[v],(posterior*field.observed)[:,None])[:,0]/support.clamp_min(1e-8)
   axes[row,ci+3].imshow(image.reshape(h,w).cpu(),vmin=0,vmax=1,cmap='magma')
   iou=record['image' if ci==0 else 'image_refined']['known_iou'];axes[row,ci+3].set_title(('Original' if ci==0 else 'Conditioned query')+f' known IoU {iou:.3f}')
  axes[row,0].set_ylabel(['Already correct','Corrected constraint','Below confidence target'][row])
for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Same frozen field; only a temporary query vector is optimized\nTarget labels are used for evaluation only. Already-correct, corrected, and remaining low-confidence source evidence.')
fig.tight_layout();fig.savefig(a.output,dpi=150)
