"""Summarize fresh full-cohort map/source runs, keeping historical references separate."""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


base = Path('output/research_consolidation')
rows = []
for scene in ['figurines', 'ramen', 'teatime', 'waldo_kitchen']:
    source = base/'source_run'/scene
    inputs = read(source/'inputs.json')
    provider = read(Path(inputs['detections'])/'provider.json')
    fit = read(source/'sets'/('lerf_'+scene)/'fit.json')
    def metrics(path):
        m = read(path)['metrics']['hierarchical_pipeline']
        return dict(semantic_iou=m['iou'], ap=m['ap'], localization=m['localization'],
                    selected_projection_iou=m['selected']['miou'])
    row = dict(scene=scene, source_views=len(provider['source_frames']),
               map=metrics(base/'map_evaluation'/scene/'evaluation.json'),
               source=metrics(source/'evaluation/evaluation.json'),
               inherited_reference=metrics(Path('output/core_scope_next104/core_update')/
                   ('lerf_'+scene)/'anchored/pipeline_calibrated/evaluation.json'),
               queries_with_no_direct_detection=[q for q,n in provider['detected_views'].items() if n == 0],
               query_count=len(provider['detected_views']), optimized_queries=len([h for h in fit['history'] if h['views']]),
               detector_seconds=provider['seconds'], set_seconds=fit['seconds'])
    row['selected_gain_over_direct_map']=row['source']['selected_projection_iou']-row['map']['selected_projection_iou']
    rows.append(row)
macro = {mode: {m:sum(r[mode][m] for r in rows)/len(rows) for m in rows[0][mode]}
         for mode in ['map','source','inherited_reference']}
result = dict(kind='fresh_automatic_text_baseline_with_frozen_development_scoring', scenes=rows,
              macro=macro, benchmark_queries=67, extra_unlabelled_queries=1,
              inherited_query_predictions=False, independent_3d_ground_truth=False,
              same_new_map_four_query=False, public_sota_comparison=False,
              optimized_rule='detector-only integration control, not all historical improvements combined')
(base/'full_execution.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(macro,indent=2))
