"""Summarize completed benchmark controls without rerunning inference or scoring."""
import json
from pathlib import Path
import numpy as np
OUT=Path('output/next72');OUT.mkdir(exist_ok=True)
nvos={}
for carrier in ['old','new']:
 for p in (Path('output/nvos_transfer_next72')/carrier).glob('*/report.json'):
  row=json.loads(p.read_text());nvos.setdefault(row['scene'],{})[carrier]=row['metrics']
def lerf_reports(root,pattern):
 result={}
 for p in Path(root).glob(pattern):
  x=json.loads(p.read_text());m=x['metrics']['hierarchical_pipeline'];result.setdefault(p.parts[2],{})[x['key']]=dict(iou=m['iou'],ap=m['ap'],selected=m['selected']['miou'])
 return result
lerf=lerf_reports('output/expanded_semantic_next72','*/old_extent/*/anchored/pipeline_calibrated/evaluation.json');extent=lerf_reports('output/expanded_extent_next72','semantic_*/*/anchored/pipeline_calibrated/evaluation.json')
matched=sorted(s for s,r in nvos.items() if set(r)=={'old','new'});summary=dict(nvos=nvos,matched_nvos_scenes=matched,nvos_full8=len(matched)==8,nvos_macro={c:{k:float(np.mean([nvos[s][c][k] for s in matched])) for k in ['iou','average_precision']} for c in ['old','new']} if matched else {},lerf=lerf,expanded_extent=extent,mainline_promoted=False,independent_test=False);(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(nvos_macro=summary['nvos_macro'],lerf=lerf,expanded_extent=extent),indent=2))
