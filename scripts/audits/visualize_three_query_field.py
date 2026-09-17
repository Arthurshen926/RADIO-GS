"""Fixed first heldout episodes: pseudo labels, unknowns and field responses."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.training.retrieval_supervision import episode_membership_targets
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--runs',nargs=2,default=['all_four_shared','without_radio_shared']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root
if a.output.exists():raise FileExistsError(a.output)
torch.set_num_threads(4);device='cuda:3';data=torch.load(root/'episodes.pt',weights_only=False);auth=json.loads((root/'responsibility.json').read_text());source=json.loads((root/'source_frames.json').read_text());sources={s['frame_idx']:s for s in source['frames']};h=auth['metadata']['feature_height'];w=auth['metadata']['feature_width'];n=auth['num_gaussians'];models=[]
assert sha256_file(root/'episodes.pt')==json.loads((root/'episodes.json').read_text())['sha256']
assert sha256_file(root/'teachers.json')==data['teacher_manifest_sha256']
assert sha256_file(root/'responsibility.json')==json.loads((root/'teachers.json').read_text())['authority_sha256']
geometry=FrozenGaussianGeometry.load(a.geometry,expected_sha256=auth['metadata']['frozen_geometry_sha256'],device=device)
identity=json.loads((root/'native_query_encoder.json').read_text())
for name in a.runs:
 report=json.loads((root/name/'report.json').read_text());path=root/name/'field.pt';assert sha256_file(path)==report['field_sha256'];model,_=load_element_retrieval_field(path,expected_sha256=report['field_sha256'],geometry=geometry,expected_query_encoder=identity,device=device);models.append(model)
fig,axes=plt.subplots(3,5,figsize=(16,9));diagnostics={'runs':a.runs,'geometry_sha256':auth['metadata']['frozen_geometry_sha256']}
with torch.no_grad():
 for row,modality in enumerate(('text','image','prompt')):
  e=next(e for e in data['episodes'] if e['modality']==modality and e['split']=='validation');vi=e['target_view'];record=auth['views'][vi];path=root/record['relative_path'];assert sha256_file(path)==record['sha256'];hit=torch.load(path,map_location=device,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weights=hit['base_weights'].float();mass=torch.zeros(h*w,device=device).scatter_add_(0,pid,weights);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),weights/mass[pid].clamp_min(1e-8),(h*w,n)).coalesce();support=torch.sparse.mm(mat,models[0].observed.float()[:,None])[:,0]
  target,known,_,_=episode_membership_targets(e,data['masks'].flatten(1).to(device),data['region_probability'].to(device),models[0].observed,support)
  src=sources[auth['frame_indices'][vi]];imagepath=Path(source['image_dir'])/src['source_file'];assert sha256_file(imagepath)==src['source_sha256'];rgb=np.array(Image.open(imagepath).resize((w,h)))
  axes[row,0].imshow(rgb);axes[row,0].set_title(modality+' target view '+str(auth['frame_indices'][vi]))
  label=np.full((h*w,3),.45);label[(known&target.bool()).cpu()]=[.15,.9,.25];label[(known&~target.bool()).cpu()]=[.9,.2,.15];axes[row,1].imshow(label.reshape(h,w,3));axes[row,1].set_title('Pseudo positive / negative / unknown')
  qi=e.get('text_index',e['query_region']);query=data['queries'][modality][qi:qi+1].float().to(device)
  stats=[]
  for ci,model in enumerate(models):
   posterior=model.retrieve(query,modality)['scores'][:,0];model_support=torch.sparse.mm(mat,model.observed.float()[:,None])[:,0];response=torch.sparse.mm(mat,(posterior*model.observed)[:,None])[:,0]/model_support.clamp_min(1e-8)
   im=axes[row,ci+2].imshow(response.reshape(h,w).cpu(),vmin=0,vmax=1,cmap='magma');axes[row,ci+2].set_title(('All four' if ci==0 else 'Without RADIO')+' posterior')
   stats.append({'known_pixel_fraction':float(known.float().mean()),'unknown_above_threshold_fraction':float(((response>.5)&~known).float().mean())})
  axes[row,4].imshow(support.reshape(h,w).cpu(),vmin=0,vmax=1,cmap='viridis');axes[row,4].set_title('Observed feature support')
  diagnostics[modality]={'query_region':e['query_region'],'text':data['queries']['queries'][qi] if modality=='text' else None,'target_frame':auth['frame_indices'][vi],'all_four_then_without_radio':stats}
for ax in axes.flat:ax.axis('off')
fig.suptitle('First heldout episode per modality; source pseudo labels, not benchmark ground truth\nUnlisted gray pixels are unknown, not negative. Posterior threshold fixed at 0.5.')
fig.tight_layout();fig.savefig(a.output,dpi=150);a.output.with_suffix('.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
