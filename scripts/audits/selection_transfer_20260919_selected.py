"""Predeclared source-only joint-output risk transfer, frozen native/final choice."""
import argparse,json,shutil
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
import numpy as np
import torch
import torch.nn.functional as F
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.scripts.eval_opengaussian_lerf_baseline import evaluate_scene
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
registration=dict(rule='Nearest source phrase in original text embedding space. Fall back to native64 iff its mean source final-minus-native exact selected-render IoU is negative. Joint source probability non-regression is an acceptance gate, not ignored. No cosine cutoff, parameter search, benchmark labels or per-scene rule changes.',source='Existing deterministic last4 source views x <=12 phrase cohort; parent model saw source views; not independent validation.',limitation='Teacher masks are visible pseudo scopes, not physical hidden-object truth. This chooses a preserved existing branch, does not introduce outside-mask negative training.',implementation_sha256=sha256_file(__file__))
(a.output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
root=Path('output/depth_opacity_pipeline_20260918')/a.scene/'reconstruction/field';sp=Path('output/source_selected_contract_20260919')/a.scene/'report.json';source=json.loads(sp.read_text());assert source['source_only'] and not source['benchmark_data_used'];ep=root/'caption_region_evidence.pt';assert sha256_file(ep)==source['evidence_sha256'];e=torch.load(ep,map_location='cpu',weights_only=False)
groups=defaultdict(list)
for row in source['observations']:groups[row['query_index']].append(row)
ids=sorted(groups);deltas=torch.tensor([[np.mean([r['metrics']['final'][metric]-r['metrics']['native'][metric] for r in groups[i]]) for metric in ['probability_iou','selected_iou']] for i in ids]);harm=deltas[:,1]<0;keys=F.normalize(e['text_embeddings'][ids].float(),dim=1)
# Source leave-one-phrase-out audit: no use of real benchmark queries or labels.
sim=keys@keys.T;sim.fill_diagonal_(-2);neighbor=sim.argmax(1);source_cv=dict(phrases=len(ids),harmful=int(harm.sum()),leave_phrase_out_accuracy=float((harm[neighbor]==harm).float().mean()),selected_delta_if_transfer=float((-deltas[harm[neighbor],1]).sum()/len(ids)),probability_delta_if_transfer=float((-deltas[harm[neighbor],0]).sum()/len(ids)))
qp=Path('output/bounded_fallback_repair_20260918')/a.scene/'text_online.encodings.pt';qbank=torch.load(qp,map_location='cpu',weights_only=False)[0];queries=F.normalize(qbank['encoding'].half().float(),dim=1);nearest=(queries@keys.T).argmax(1);choices={text:('native_compact_calibrated' if harm[nearest[i]] else 'final') for i,text in enumerate(qbank['texts'])}
decisions=[dict(query=text,branch=choices[text],source_phrase=e['phrases'][ids[int(nearest[i])]],source_probability_delta=float(deltas[nearest[i],0]),source_selected_delta=float(deltas[nearest[i],1]),similarity=float(queries[i]@keys[nearest[i]])) for i,text in enumerate(qbank['texts'])]
(a.output/'decisions.json').write_text(json.dumps(dict(source_sha256=sha256_file(sp),query_sha256=sha256_file(qp),source_cv=source_cv,decisions=decisions),indent=2)+'\n')
# Predictions assembled exclusively from previously generated exact branch outputs, before evaluator reads GT.
pred=a.output/'selected_predictions';pred.mkdir();previous=Path('output/lerf_selected_ladder_20260919')/a.scene
for file in sorted((previous/'final').glob('*.png')):
 matches=[text for text in choices if file.name.endswith('_'+text+'.png')];assert len(matches)==1,(file,matches);q=matches[0];shutil.copyfile(previous/choices[q]/file.name,pred/file.name)
manifest={f.name:sha256_file(f) for f in sorted(pred.glob('*.png'))};(a.output/'prediction_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
gt=Path('output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/eval_gt')/a.scene/'gt';metric=asdict(evaluate_scene(gt,pred,a.scene,threshold=10));assert metric['missing']==0
# Existing exact per-observation native-grid measurements; not used in decisions.
raw=json.loads((Path('output/lerf_native_membership_20260919')/a.scene/'report.json').read_text());rows=[dict(frame=r['frame'],query=r['query'],branch=choices[r['query']],iou=r['membership_iou'][choices[r['query']]]) for r in raw['observations']]
base=json.loads((previous/'report.json').read_text())['results']['final'];report=dict(registration,scene=a.scene,source_cv=source_cv,decision_sha256=sha256_file(a.output/'decisions.json'),prediction_manifest_sha256=sha256_file(a.output/'prediction_manifest.json'),selected_baseline=base,selected_candidate=metric,probability_baseline=float(raw['macro_membership_iou']['final']),probability_candidate=float(np.mean([r['iou'] for r in rows])),observations=rows,fallback_queries=[q for q,v in choices.items() if v!='final'],geometry_unchanged=True,query_encoder_unchanged=True,official_threshold_unchanged=True,independent_test=False,promotion=False)
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['observations']},indent=2))
