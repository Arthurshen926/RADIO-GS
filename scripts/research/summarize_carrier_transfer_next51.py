"""Separate annotation-assisted transfer from source-only method results."""
import json
from pathlib import Path
from statistics import mean
out=Path('output/carrier_transfer_next51')
transfer={}
for p in sorted(out.glob('*/*/diagnostic.json')):
    d=json.loads(p.read_text());objects=d['selected']['objects'];seen={r['query'] for r in objects if r['frame'] in d['fit_frames']}
    groups={
        'fit':[r for r in objects if r['frame'] in d['fit_frames']],
        'held_all':[r for r in objects if r['frame'] in d['held_frames']],
        'held_seen_queries':[r for r in objects if r['frame'] in d['held_frames'] and r['query'] in seen],
        'held_unseen_queries':[r for r in objects if r['frame'] in d['held_frames'] and r['query'] not in seen]}
    transfer.setdefault(d['geometry'],{})[d['scene']]={k:dict(count=len(v),iou=mean(r['iou'] for r in v) if v else None) for k,v in groups.items()}
methods={}
for root in ['output/evidence_balance_next51','output/refined_selection_next51']:
    for p in sorted(Path(root).glob('*/*/anchored/pipeline_calibrated/evaluation.json')):
        m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];methods.setdefault(p.parts[1]+'/'+p.parts[2],{})[p.parts[3]]=dict(iou=m['iou'],ap=m['ap'],selected=m['selected']['miou'])
summary=dict(annotation_assisted_diagnostic_not_method=transfer,source_only_method_results=methods,default_changed=False)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
