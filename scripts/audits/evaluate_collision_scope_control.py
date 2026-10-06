"""Compare frozen query-relative intervention on source collision masks only."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.query_relative_scope import install_query_relative_scope_control
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';base=Path('output/current_recipe_audit_20260918')/a.scene;out=base/'collision_scope_control.json';assert not out.exists();d=json.loads((base/'source_collisions_detailed.json').read_text());fp=root/'counterfactual_extent_benefit_admission/field.pt';assert sha256_file(fp)==d['field_sha256'];payload=torch.load(fp,weights_only=False);field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False);ep=root/'caption_region_evidence.pt';assert sha256_file(ep)==d['evidence_sha256'];e=torch.load(ep,weights_only=False);auth=json.loads((root/'responsibility.json').read_text());info=json.loads((root/'teachers.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];tasks={}
for pair in d['identical_membership_coobserved_pairs']:
 if pair['minimum_source_mask_iou']>=.5:continue
 for side in ['left','right']:
  for row in pair['coobserved']:tasks[pair[side]['query'],row['frame']]=pair[side]['phrase']
ids=sorted({i for i,f in tasks});q=e['text_embeddings'][ids].float().to(a.device);rows=[]
with torch.no_grad():
 baseline=field.retrieve(q,'text')['scores'];undo=install_query_relative_scope_control(field)
 try:control=field.retrieve(q,'text')['scores']
 finally:undo()
 pixels=e['masks'][0].numel();n=len(field.latent)
 for frame in sorted({f for i,f in tasks}):
  vi=info['frames'].index(frame);rec=auth['views'][vi];hp=root/rec['relative_path'];assert sha256_file(hp)==rec['sha256'];hit=torch.load(hp,map_location=a.device,weights_only=False);pid,gid,w=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float();full=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,w);keep=field.observed[gid];pid,gid,w=pid[keep],gid[keep],w[keep];mass=torch.zeros_like(full).scatter_add_(0,pid,w);valid=(full>1e-8)&(mass/full.clamp_min(1e-8)>.5);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce();outputs=[torch.sparse.mm(mat,v) for v in [baseline,control]]
  for qi in ids:
   if (qi,frame) not in tasks:continue
   rr=[i for i,r in enumerate(e['records']) if r['frame']==frame and qi in r['phrase_indices'] and r['confidence']>=.7 and int(e['masks'][i].sum())>=20];target=e['masks'][rr].bool().any(0).flatten().to(a.device);row=dict(query=qi,phrase=tasks[qi,frame],frame=frame)
   for name,score in zip(['retained','query_relative'],outputs):
    pred=(score[:,ids.index(qi)]>.5)&valid;tp=int((pred&target).sum());row[name]=dict(iou=tp/max(int((pred|target).sum()),1),false_positive_pixels=int((pred&~target).sum()),false_negative_pixels=int((~pred&target).sum()))
   rows.append(row)
report=dict(source_only=True,benchmark_data_used=False,in_sample_selected_failure_cases=True,independent_validation=False,field_sha256=sha256_file(fp),evidence_sha256=sha256_file(ep),observations=rows,macro={name:sum(r[name]['iou'] for r in rows)/len(rows) for name in ['retained','query_relative']},implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/querying/query_relative_scope.py'))
out.write_text(json.dumps(report,indent=2)+'\n');print(a.scene,report['macro'],len(rows))
