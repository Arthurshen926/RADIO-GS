import json,torch
from pathlib import Path
rows=[];decisions=[];load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 key='lerf_'+scene;base=Path('output/core_scope_next104/core_update')/key;out=Path('output/literal_update_next105')/key;b=load(base/'typed_readout.pt');c=load(out/'typed_readout.pt')
 def m(p):return json.loads((p/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']['selected']['miou']
 rows.append(dict(scene=scene,baseline=m(base),candidate=m(out),semantic_exact=torch.equal(b['scores'],c['scores']),selection_exact=torch.equal(b['selection_scores'],c['selection_scores'])))
 decisions.extend(json.loads((out.parent/(key+'_decisions.json')).read_text()))
counts=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 for r in json.loads((Path('output/literal_candidates_next105')/('lerf_'+scene)/'proposals.json').read_text()):counts.append(len(r['candidates']))
report=dict(scenes=rows,macro=sum(r['candidate'] for r in rows)/4,updated_queries=[r['query'] for r in decisions if r['update']],source_decisions=decisions,pairs=sum(counts),regions_verified=2*sum(counts),new_tracking_runs=0,new_scope_fits=0,unchanged_output_evaluations=4)
Path('output/next105/summary.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['source_decisions','scenes']})
