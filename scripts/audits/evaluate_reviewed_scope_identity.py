"""Separate native text identity, extent readout and reviewed-reference identity."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file

parser=argparse.ArgumentParser();parser.add_argument("--field-root",type=Path);parser.add_argument("--run",default="counterfactual_extent_benefit_admission");parser.add_argument("--output",type=Path,default=Path("output/expert_plan_20260918/reviewed_bag_identity.json"));args=parser.parse_args()
torch.set_num_threads(4)
review_path=Path('docs/reviews/2026-09-18-bag-scope-track.json')
review=json.loads(review_path.read_text())
root=Path('output/pixel_evidence_rebuild_20260917')/review['scene']/'field'
ep=root/'caption_region_evidence.pt'
assert sha256_file(ep)==json.loads(Path(review['candidate_manifest']).read_text())['evidence_sha256']
e=torch.load(ep,weights_only=False)
fp=(args.field_root or root)/args.run/'field.pt'
assert sha256_file(fp)==json.loads(fp.with_name('report.json').read_text())['field_sha256']
payload=torch.load(fp,weights_only=False)
field=ElementRetrievalField(**payload['architecture']);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
assert sha256_file(root/'responsibility.json')==info['authority_sha256']
pixels=e['masks'][0].numel();rows=[]
with torch.no_grad():
 q=e['text_embeddings'][e['phrases'].index(review['query'])][None].float()
 native=(field.readout_log_scale['text'].clamp(-4,6).exp()*(field.native_cosine(q)-field.native_query_thresholds['text'](q))).sigmoid()[:,0]
 complete=field.retrieve(q,'text')['scores'][:,0]
 for observation in review['observations']:
  frame=observation['frame'];target=e['masks'][observation['region']].bool().flatten()
  record=auth['views'][info['frames'].index(frame)];hp=root/record['relative_path'];assert sha256_file(hp)==record['sha256']
  hit=torch.load(hp,weights_only=False);pid,gid,weight=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float()
  mass=torch.zeros(pixels).scatter_add_(0,pid,weight)
  mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(pixels,len(field.latent))).coalesce()
  reference_ids=[r['region'] for r in review['observations'] if r['frame']!=frame]
  iq=torch.nn.functional.normalize(e['native_descriptors'][reference_ids].float(),dim=-1).mean(0,keepdim=True)
  image=field.retrieve(iq,'image')['scores'][:,0]
  row=dict(frame=frame,reference_regions=reference_ids)
  for name,value in [('native_text',native),('complete_text',complete),('reviewed_reference_image',image)]:
   known=field.observed_for('image' if name=='reviewed_reference_image' else 'text')
   support=torch.sparse.mm(mat,known.float()[:,None])[:,0]
   score=torch.sparse.mm(mat,(value*known)[:,None])[:,0]/support.clamp_min(1e-8)
   pred=(score>.5)&(support>.5);tp=int((pred&target).sum())
   row[name]=dict(iou=tp/max(int((pred|target).sum()),1),precision=tp/max(int(pred.sum()),1),recall=tp/max(int(target.sum()),1))
  rows.append(row)
report=dict(schema='reviewed_scope_identity.v1',source_only=True,benchmark_data_used=False,reviewer='assistant_visual_review',training=False,observations=rows,
 macro={name:sum(r[name]['iou'] for r in rows)/len(rows) for name in ['native_text','complete_text','reviewed_reference_image']},
 review_sha256=sha256_file(review_path),evidence_sha256=sha256_file(ep),field_path=str(fp),field_sha256=sha256_file(fp),implementation_sha256=sha256_file(__file__),
 limitations='Selected source object; reference teacher crops selected by assistant review, target excluded from reference pooling. Existing map saw source views. Image query has more identity information than text: intervention diagnostic, not equal-information benchmark comparison.')
out=args.output;assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(report['macro'])
