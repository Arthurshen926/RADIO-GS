"""Separate full-cohort task recipes from resolution and representation controls."""
import json
from pathlib import Path


def mean(rows):return {k:sum(r[k] for r in rows)/len(rows) for k in rows[0]} if rows else {}


def main():
 result={'development_only':True,'units':'percent; equal-scene mean','methods':{}}
 bases=['scope_composition_matched_next38','scope_composition_local_next38','scope_composition_native_next38','scope_written_matched_next38','category_competition_next38','high_resolution_scopes_next38','taskwise_scope_composition_next38']
 for base in bases:
  root=Path('output')/base;rows=[]
  for path in sorted(root.glob('*/anchored/*/evaluation.json')):
   data=json.loads(path.read_text());metrics={}
   for arm,v in data['metrics'].items():
    metrics[arm]=({k:100*v[k]['miou'] for k in ['19','15','10']} if '19' in v else {k:100*v[k] for k in ['iou','ap','localization']}|{'selected_iou':100*v['selected']['miou']})
   rows.append(dict(key=data['key'],metrics=metrics,path=str(path)))
  output={'scenes':rows,'cohorts':{}}
  for cohort in ['lerf','scannet']:
   selected=[r for r in rows if r['key'].startswith(cohort)];output['cohorts'][cohort]={'count':len(selected),'macro':{arm:mean([r['metrics'][arm] for r in selected]) for arm in ['retained_pipeline','hierarchical_pipeline']}}
  for kind,pattern in [('image','*/reference_image_evaluation.json'),('nvos','*/evaluation_*/evaluation.json')]:
   selected=[]
   for path in sorted(root.glob(pattern)):
    data=json.loads(path.read_text());metrics=data['macro' if kind=='image' else 'metrics'];selected.append(dict(key=data.get('key',data.get('scene')),path=str(path),metrics={arm:{k:100*v[k] for k in ['iou','average_precision']} for arm,v in metrics.items() if arm in ['retained','trained']}))
   output['cohorts'][kind]={'count':len(selected),'scenes':selected,'macro':{arm:mean([r['metrics'][arm] for r in selected]) for arm in ['retained','trained']}}
  execution=root/'execution.json'
  if execution.exists():output['execution']=json.loads(execution.read_text())
  result['methods'][base]=output
  print(base,{k:(v['count'],v['macro'].get('hierarchical_pipeline',v['macro'].get('trained'))) for k,v in output['cohorts'].items()})
 result['resolution_evidence']={p.parent.name:json.loads(p.read_text()) for p in Path('output/high_resolution_scope_evidence_next38').glob('*/evidence.json')}
 result['resolution_scope_diagnostics']={p.parent.name:json.loads(p.read_text())['metrics'] for p in Path('output/high_resolution_scopes_next38').glob('*/hypothesis_retrieval.json')}
 result['category_source_validation']={p.parent.name:json.loads(p.read_text()) for p in Path('output/category_competition_next38').glob('*/training.json')}
 Path('output/taskwise_scope_composition_next38/summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
