"""Bind two preregistered source risk controls and view-disjoint gate validation."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/selection_transfer_20260919');out=root/'summary.json';assert not out.exists();results={}
for scene in ['figurines','ramen']:
 fitpath=Path('output/source_selected_contract_20260919')/scene/'report.json';fit=json.loads(fitpath.read_text());holdpath=root/f'source_holdout_{scene}'/'report.json';hold=json.loads(holdpath.read_text());assert not {r['frame'] for r in fit['observations']}&{r['frame'] for r in hold['observations']}
 ep=Path('output/depth_opacity_pipeline_20260918')/scene/'reconstruction/field/caption_region_evidence.pt';assert sha256_file(ep)==fit['evidence_sha256']==hold['evidence_sha256'];e=torch.load(ep,map_location='cpu',weights_only=False);groups=defaultdict(list)
 for r in fit['observations']:groups[r['query_index']].append(r)
 ids=sorted(groups);vectors=F.normalize(e['text_embeddings'].float(),dim=1);delta=torch.tensor([[np.mean([r['metrics']['final'][m]-r['metrics']['native'][m] for r in groups[i]]) for m in ['probability_iou','selected_iou']] for i in ids]);branches={}
 for arm in ['joint','selected']:
  harms=(delta<0).all(1) if arm=='joint' else delta[:,1]<0;rows=[]
  for r in hold['observations']:
   ni=int((vectors[r['query_index']]@vectors[ids].T).argmax());branch='native' if harms[ni] else 'final';rows.append(dict(frame=r['frame'],phrase=r['phrase'],nearest_phrase=e['phrases'][ids[ni]],query_seen_in_risk_fit=r['query_index'] in ids,branch=branch,metrics=r['metrics'][branch]))
  names=['probability_iou','selected_iou','selected_precision','selected_recall'];candidate={m:float(np.mean([r['metrics'][m] for r in rows])) for m in names};baseline=hold['macro']['final'];accepted=candidate['probability_iou']>=baseline['probability_iou']-1e-8 and candidate['selected_iou']>baseline['selected_iou']+1e-8
  path=root/(scene if arm=='joint' else f'selected_risk_{scene}')/'report.json';report=json.loads(path.read_text());branches[arm]=dict(risk_fit_phrases=len(ids),risk_positive_phrases=int(harms.sum()),gate_view_holdout_observations=len(rows),gate_view_holdout_fallbacks=sum(r['branch']=='native' for r in rows),holdout_baseline=baseline,holdout_candidate=candidate,source_gate_acceptance=accepted,source_holdout_rows=rows,benchmark_probability_baseline=report['probability_baseline'],benchmark_probability_candidate=report['probability_candidate'],benchmark_selected_baseline=report['selected_baseline']['miou'],benchmark_selected_candidate=report['selected_candidate']['miou'],benchmark_count=report['selected_candidate']['count'],benchmark_fallbacks=report['fallback_queries'],report_sha256=sha256_file(path))
 results[scene]=branches
result=dict(schema='selection_transfer_bounded_control.v1',rule_choice_before_target_evaluation=True,benchmark_gt_not_used_for_rule=True,source_teacher_masks_not_physical_gt=True,gate_holdout_frames_disjoint=True,parent_model_saw_holdout_frames=True,independent_test=False,promotion=False,results=results,implementation_sha256=sha256_file(__file__))
out.write_text(json.dumps(result,indent=2)+'\n')
for scene,arms in results.items():
 for arm,r in arms.items(): print(scene,arm,{k:v for k,v in r.items() if k not in ['source_holdout_rows','report_sha256']})
