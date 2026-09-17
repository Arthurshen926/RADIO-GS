"""Measure what actual source membership losses can update in the protected field.

No optimizer step, target benchmark query, geometry update or teacher claim.
Auxiliary teacher gradients are outside this audit; inspect training receipts.
"""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.training.retrieval_supervision import episode_membership_targets,membership_loss
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
assert not a.output.exists();root=Path('output/lerf_ropefix_multiscale_pipeline_20260917')/a.scene/'field'
fp=root/'ropefix96_protected/field.pt';fr=json.loads(fp.with_name('report.json').read_text());assert sha256_file(fp)==fr['field_sha256'] and fr['protect_native_semantic'] and fr['freeze_native_decoder']
payload=torch.load(fp,weights_only=False);field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.native_decoder.requires_grad_(False)
mask=torch.ones_like(field.latent);mask[:,:field.native_semantic_dim]=0;field.latent.register_hook(lambda g:g*mask)
ep_path=root/'episodes.pt';assert sha256_file(ep_path)==json.loads(ep_path.with_suffix('.json').read_text())['sha256']
ep=torch.load(ep_path,weights_only=False);gtpath=root/'grounded_text_supervision.pt';assert sha256_file(gtpath)==fr['grounded_text_supervision_sha256'];ground=torch.load(gtpath,weights_only=False)
info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert ep['teacher_manifest_sha256']==sha256_file(root/'teachers.json');assert sha256_file(root/'responsibility.json')==info['authority_sha256'];assert ep['geometry_binding']==payload['geometry_binding']
masks=ep['masks'].flatten(1).to(a.device);rp=ep['region_probability'].float().to(a.device);rows=[]
for modality in ('text','image','prompt'):
 e=next(iter(ground['tasks'])) if modality=='text' else next(e for e in ep['episodes'] if e['modality']==modality and e['split']=='train')
 vi=e['target_view'];hitrec=auth['views'][vi];hp=root/hitrec['relative_path'];assert sha256_file(hp)==hitrec['sha256'];hit=torch.load(hp,map_location=a.device,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();w=hit['base_weights'].float();pixels=masks.shape[1]
 full=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,w);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/full[pid].clamp_min(1e-8),(pixels,len(field.latent))).coalesce();support=torch.sparse.mm(mat,field.observed.float()[:,None])[:,0]
 qi=e.get('text_index',e.get('query_region'));query=ep['queries'][modality][qi:qi+1].float().to(a.device);probs=field.retrieve(query,modality)['scores'][:,0];render=torch.sparse.mm(mat,(probs*field.observed)[:,None])[:,0]/support.clamp_min(1e-8)
 if modality=='text':
  ti=e['grounded_text_task'];pos=ground['positive_pixels'][ti].to(a.device);neg=ground['negative_pixels'][ti].to(a.device);rpos=ground['positive_elements'][qi].to(a.device);rneg=ground['negative_elements'][qi].to(a.device)
  target=pos.float();known=(pos^neg)&(support>.5);rt=rpos.float();rk=(rpos^rneg)&field.observed&ep['seen'][vi].to(a.device)
 else:target,known,rt,rk=episode_membership_targets(e,masks,rp,field.observed,support)
 loss=membership_loss(render,target,known)+membership_loss(probs,rt,rk)
 params={k:v for k,v in field.named_parameters() if v.requires_grad};grads=torch.autograd.grad(loss,list(params.values()),allow_unused=True);norms={k:float(g.norm()) if g is not None else 0. for k,g in zip(params,grads)};latent=grads[list(params).index('latent')]
 rows.append(dict(modality=modality,source_query_index=qi,target_view=vi,known_pixels=int(known.sum()),known_elements=int(rk.sum()),loss=float(loss),native_coordinate_gradient_norm=float(latent[:,:field.native_semantic_dim].norm()),structural_coordinate_gradient_norm=float(latent[:,field.native_semantic_dim:].norm()),parameter_gradient_norms=norms))
print(rows,flush=True)
result=dict(schema='protected_source_membership_gradient.v1',source_only=True,benchmark_data_used=False,optimizer_steps=0,auxiliary_teacher_losses_included=False,field_sha256=fr['field_sha256'],episode_sha256=sha256_file(ep_path),grounded_supervision_sha256=sha256_file(gtpath),implementation_sha256=sha256_file(__file__),observations=rows)
a.output.write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(__file__,a.output.with_suffix('.source.py'))
