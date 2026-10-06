"""Summarize source-response and query-independent latent transport separately."""
import json
from pathlib import Path
from statistics import mean
summary={}
for name,root in [('responses',Path('output/unified_response_next53')),('latent',Path('output/latent_transport_next53'))]:
 summary[name]={}
 for mode in ['direct','transport']:
  rows={}
  for p in (root/mode).glob('*/anchored/pipeline_calibrated/evaluation.json'):
   m=json.loads(p.read_text())['metrics'];rows[p.parts[3]]=dict(metrics=m,fit=json.loads((p.parents[2]/'fit.json').read_text()))
  if rows:summary[name][mode]=dict(scenes=rows,macro={k:mean(r['metrics']['hierarchical_pipeline'][k] for r in rows.values()) for k in ['iou','ap','localization']},macro_selected=mean(r['metrics']['hierarchical_pipeline']['selected']['miou'] for r in rows.values()))
 native={}
 for p in root.glob('*/*/anchored/evaluation.json'):
  native.setdefault(p.parts[2],{})[p.parts[3]]=json.loads(p.read_text())['metrics']['native64']
 if native:summary[name]['native_only']=native
summary.update(development_only=True,default_changed=False,full_four_teacher_retraining=False)
Path('output/unified_response_next53/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:{a:v['macro'] for a,v in r.items() if isinstance(v,dict) and 'macro' in v} for k,r in summary.items() if isinstance(r,dict)},indent=2))
