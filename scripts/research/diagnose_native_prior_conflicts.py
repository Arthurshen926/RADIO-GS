"""Evaluation-only attribution of native-prior corrections, never inference labels."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from plyfile import PlyData

ROOT=Path('output/native_evidence_contribution')
rows={};total=Counter();wrong_transitions=Counter()
for config in sorted(Path('configs/pipeline/scannet_native_complete').glob('*.scene.json')):
 name=config.name.removesuffix('.scene.json')
 evaluation=json.loads(Path(f'configs/consolidated/scannet_native/{name}.evaluation.json').read_text())
 truth=np.asarray(PlyData.read(evaluation['ground_truth'])['vertex'].data['label'])
 valid=np.isin(truth,evaluation['splits']['19'])
 native=np.load(Path('output/system_v2/scannet',name,'evaluation/source_point_predictions.npz'))['19']
 uniform=np.load(ROOT/name/'uniform_prior/evaluation/source_point_predictions.npz')['19']
 helped=valid & (native==truth) & (uniform!=truth)
 harmed=valid & (native!=truth) & (uniform==truth)
 changed=valid & (native!=uniform)
 record=dict(valid_points=int(valid.sum()),changed_points=int(changed.sum()),corrected_by_native=int(helped.sum()),spoiled_by_native=int(harmed.sum()),both_wrong_changed=int((changed & ~helped & ~harmed).sum()))
 total.update(record)
 metric=json.loads(Path('output/system_v2/scannet',name,'evaluation/report.json').read_text())['point_metrics']['source']['19']
 names={int(k):v['name'] for k,v in metric['per_class'].items()}
 classes={names[c]:dict(corrected=int((helped & (truth==c)).sum()),spoiled=int((harmed & (truth==c)).sum()),gt_points=int((truth==c).sum())) for c in evaluation['splits']['19'] if (truth==c).any()}
 for gt,pred in zip(truth[harmed],native[harmed]):wrong_transitions[(names[int(gt)],names[int(pred)])]+=1
 rows[name]=dict(**record,classes=classes)
summary=dict(scope='evaluation-only19class points; pooled counts are not scene-macro IoU',scenes=rows,pooled_counts=dict(total),top_harmful_changes=[dict(truth=gt,native_prediction=pred,points=n) for (gt,pred),n in wrong_transitions.most_common(12)])
(ROOT/'prior_conflicts.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='scenes'},indent=2))
