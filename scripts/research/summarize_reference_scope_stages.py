"""Summarize fixed eight-query source/target scope diagnostics."""
import json
from pathlib import Path
root=Path('output/reference_scope_stages');rows=[]
for scene in ['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex']:
    p=root/scene;d=json.loads((p/'diagnosis.json').read_text());d['reference_only_target']=json.loads((p/'reference_only/evaluation/evaluation.json').read_text())['metrics'];rows.append(d)
result=dict(scenes=rows,macro=dict(reference_only_target_iou=sum(d['reference_only_target']['iou'] for d in rows)/8,full_tracks_target_iou=sum(d['target']['response']['iou'] for d in rows)/8,source_response_agreement=sum(d['source_macro']['response_vs_track']['iou'] for d in rows)/8,source_subset_agreement=sum(d['source_macro']['subset_vs_track']['iou'] for d in rows)/8),development_only=True,source_agreement_is_not_accuracy=True,target_rgb_used=False,new_tracking=False,default_changed=False)
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(result['macro'])
