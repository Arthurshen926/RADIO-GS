"""Audit source phrase-split contamination; produce conservative family folds."""
import argparse,json
from pathlib import Path
from itertools import combinations
import torch
from radio_gs.training.scope_family_split import scope_family_split
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists();path=a.root/'caption_region_evidence.pt';assert sha256_file(path)==json.loads(path.with_suffix('.json').read_text())['sha256'];e=torch.load(path,weights_only=False);assert e['source_only'] and not e['benchmark_data_used'];views={}
for ri,r in enumerate(e['records']):
 if r['confidence']<.7 or int(e['masks'][ri].sum())<20:continue
 masks=views.setdefault(r['frame'],{})
 for qi in set(r['phrase_indices']):masks[qi]=masks.get(qi,torch.zeros_like(e['masks'][ri],dtype=torch.bool))|e['masks'][ri].bool()
ids=sorted({q for masks in views.values() for q in masks});edges={}
for frame,masks in views.items():
 qs=sorted(masks);x=torch.stack([masks[q].flatten() for q in qs]).float();inter=x@x.T;areas=x.sum(1);iou=inter/(areas[:,None]+areas[None]-inter).clamp_min(1);contain=inter/torch.minimum(areas[:,None],areas[None]).clamp_min(1)
 for i,j in combinations(range(len(qs)),2):
  if iou[i,j]>=.5 or contain[i,j]>=.8:
   key=(qs[i],qs[j]);edges.setdefault(key,[]).append(dict(frame=frame,iou=float(iou[i,j]),containment=float(contain[i,j])))
plan=scope_family_split(ids,list(edges));admission=a.root/'counterfactual_extent_benefit_admission/report.json';old=None
if admission.exists():
 r=json.loads(admission.read_text());held=set(r['heldout_phrases']);validation={q for q in ids if e['phrases'][q] in held};training=set(ids)-validation;cross=[(x,y) for x,y in edges if (x in validation and y in training) or (y in validation and x in training)];leaked={q for pair in cross for q in pair if q in validation};old=dict(validation_queries=len(validation),validation_queries_with_train_overlap=len(leaked),cross_split_evidence_edges=len(cross),admission_report_sha256=sha256_file(admission))
report=dict(schema='source_scope_family_leakage.v1',source_only=True,benchmark_data_used=False,physical_identity_verified=False,folds_are_full_pipeline_independent=False,evidence_sha256=sha256_file(path),queries=[dict(query=q,phrase=e['phrases'][q]) for q in ids],edges=[dict(left=x,right=y,observations=rr) for (x,y),rr in sorted(edges.items())],split=plan,old_phrase_split=old,implementation_sha256=sha256_file(__file__),split_implementation_sha256=sha256_file('radio_gs/training/scope_family_split.py'),limitation='Conservative contamination families based on same-view IoU>=.5 or containment>=.8. Could overmerge broad context; not manual instance identities. Parent field/heads are not retrained by this audit; folds alone do not establish independent validation.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print('families',len(plan['families']),'sizes',[len(f) for f in plan['families']],'old',old)
