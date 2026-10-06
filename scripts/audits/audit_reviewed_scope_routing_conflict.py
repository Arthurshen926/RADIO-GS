"""Source-only fixed-head identity collision and positive-gradient saturation audit."""
import json,torch
from pathlib import Path
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
torch.set_num_threads(4)
r=Path('output/v2_scope_rebuild_20260918/figurines/field');fp=r/'counterfactual_extent_benefit_admission/field.pt';p=torch.load(fp,weights_only=False);m=ElementRetrievalField(**p['architecture']);m.load_state_dict(p['state_dict']);m.requires_grad_(False);h=m.grounded_transfer;e=torch.load(r/'caption_region_evidence.pt',weights_only=False);qi=e['phrases'].index('plastic bag');q=e['text_embeddings'][qi:qi+1].float();g,accepted,score=h.route(q);idx=int(g)+h.identity_selection['trusted_concept_start'];z=h.features(m.latent)
raw=(F.linear(z,h.decoder.weight[idx:idx+1],h.decoder.bias[idx:idx+1])+F.linear(h.residual[1](h.residual[0](z)),h.residual[2].weight[idx:idx+1],h.residual[2].bias[idx:idx+1])).detach()[:,0]
info=json.loads((r/'teachers.json').read_text());auth=json.loads((r/'responsibility.json').read_text());rows=[]
for ob in json.load(open('docs/reviews/2026-09-18-bag-scope-track.json'))['observations']:
 frame=ob['frame'];ids=[i for i,rec in enumerate(e['records']) if rec['frame']==frame and qi in rec['phrase_indices']];target=torch.zeros_like(e['masks'][0],dtype=torch.bool)
 for i in ids:target|=e['masks'][i].bool()
 if not ids:continue
 rec=auth['views'][info['frames'].index(frame)];hit=torch.load(r/rec['relative_path'],weights_only=False);pid,gid,w=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float();keep=m.observed[gid];pid,gid,w=pid[keep],gid[keep],w[keep];positive=target.flatten()[pid];weights=w[positive];values=raw[gid[positive]]
 rows.append(dict(frame=frame,source_mask_ids=ids,positive_mass_clip_low=float(weights[values<=1e-6].sum()/weights.sum()),positive_mass_clip_high=float(weights[values>=1-1e-6].sum()/weights.sum()),positive_mass_under_threshold=float(weights[values<=.5].sum()/weights.sum()),weighted_mean_raw=float((weights*values).sum()/weights.sum())))
groups=json.loads((r/'source_phrase_groups.json').read_text())['groups'];group=groups[int(g)];prototype_qi=group['source_query_index'];prototype=e['text_embeddings'][prototype_qi:prototype_qi+1].float();conflicts=[]
with torch.no_grad():
 actual=m.retrieve(torch.cat([q,prototype]),'text')['scores'];maximum_difference=float((actual[:,0]-actual[:,1]).abs().max())
for row in rows:
 frame=row['frame'];masks=[]
 for query_index in [qi,prototype_qi]:
  members=[i for i,rec in enumerate(e['records']) if rec['frame']==frame and query_index in rec['phrase_indices']]
  masks.append(e['masks'][members].bool().any(0) if members else torch.zeros_like(e['masks'][0],dtype=torch.bool))
 left,right=masks;conflicts.append(dict(frame=frame,query_positive_pixels=int(left.sum()),prototype_positive_pixels=int(right.sum()),intersection=int((left&right).sum()),iou=float((left&right).sum()/(left|right).sum().clamp_min(1))))
print(rows,group['phrase'],conflicts,maximum_difference)
out=dict(field_sha256=sha256_file(fp),query='plastic bag',routed_phrase=group['phrase'],routed_phrase_views=len(group['views']),same_view_conflicts=conflicts,maximum_query_probability_difference=maximum_difference,concept_group=int(g),concept_accepted=bool(accepted),selected_extent=idx,source_observations=rows,source_only=True,review_used_to_choose_diagnostic_query=True,training=False,implementation_sha256=sha256_file(__file__),evidence_sha256=sha256_file(r/'caption_region_evidence.pt'))
destination=Path('output/expert_plan_20260918/bag_routing_conflict.json');assert not destination.exists();destination.write_text(json.dumps(out,indent=2)+'\n')
