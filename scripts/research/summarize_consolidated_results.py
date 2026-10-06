"""Reaggregate named existing result families, never mix them into one model score.
Reads evaluation JSON only; does not run inference, change predictions or tune rules.
"""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--registry',default='configs/recipes/research_consolidation.json');p.add_argument('--output',default='output/research_consolidation/results.json');a=p.parse_args()
registry=json.loads(Path(a.registry).read_text());results={}
for name,family in registry['result_families'].items():
 rows=[]
 for item in family['reports']:
  data=json.loads(Path(item['path']).read_text());values={}
  for metric,keys in family['metric_paths'].items():
   value=data
   for key in keys:value=value[key]
   values[metric]=float(value)
  rows.append(dict(unit=item['unit'],path=item['path'],metrics=values))
 results[name]=dict(status=family['status'],protocol=family['protocol'],count=len(rows),macro={m:sum(x['metrics'][m] for x in rows)/len(rows) for m in family['metric_paths']},reports=rows)
result=dict(kind='reaggregation_of_existing_evaluations_not_prediction_rescoring',single_unified_model=False,independent_test=False,result_families=results)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
for name,r in results.items():print(name,r['count'],r['macro'])
