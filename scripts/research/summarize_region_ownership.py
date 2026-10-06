"""Complete cohort comparisons; scene-macro aggregation, no winner selection."""
import json
from pathlib import Path
LERF=['ramen','figurines','teatime','waldo_kitchen']
SCAN=[r['scene'] for r in json.loads(Path('configs/consolidated/scannet/cohort.json').read_text())['scenes']]
def read(p):return json.loads(Path(p).read_text())
def lerf(p):
 d=read(p)['metrics']['hierarchical_pipeline']
 return dict(iou=d['iou'],ap=d['ap'],localization=d['localization'],selected_projection_iou=d['selected']['miou'])
def mean(rows):return {k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}
result=dict(lerf={},scannet={},all_evaluation_scenes_are_development=True,ground_truth_used_for_training=False,query_coordinates='frozen native encoder',source_evidence='fixed admitted LERF masks and fixed176 ScanNet category views',promotion='none: all three candidates regress selected projection and ScanNet versus the retained global recipe')
for arm in ['baseline','finest','early_mean','late_logmeanexp']:
 values={}
 for n in LERF:
  root=Path('output/system_map_rebuild')/n if arm=='baseline' else (Path('output/native_region_ownership')/n if arm=='finest' else Path('output/native_scope_pair')/n/arm)
  values[n]=dict(map=lerf(root/('map_evaluation/evaluation.json' if arm=='baseline' else 'evaluation/evaluation.json')),source=lerf(root/'source_evaluation/evaluation.json'))
 result['lerf'][arm]=dict(scenes=values,macro={k:mean([v[k] for v in values.values()]) for k in ['map','source']})
 values={}
 for n in SCAN:
  path=(Path('output/native_category_transfer')/n/'public/category/evaluation/report.json' if arm=='baseline' else (Path('output/native_region_ownership')/n/'evaluation/report.json' if arm=='finest' else Path('output/native_scope_pair')/n/arm/'evaluation/report.json'))
  d=read(path)['point_metrics'];values[n]={k:{s:d[k][s]['miou'] for s in ['19','15','10']} for k in ['map','source']}
 result['scannet'][arm]=dict(scenes=values,macro={k:mean([v[k] for v in values.values()]) for k in ['map','source']})
result['deployment_files']={n:{k:Path(p).stat().st_size for k,p in dict(primary=f'output/system_map_rebuild/{n}/complete/native_field.pt',fine=f'output/native_region_ownership/{n}/map/native_field.pt').items()} for n in LERF}
Path('output/native_region_ownership/summary.json').write_text(json.dumps(result,indent=2)+'\n')
for dataset in ['lerf','scannet']:
 for arm,d in result[dataset].items():print(dataset,arm,d['macro'])
