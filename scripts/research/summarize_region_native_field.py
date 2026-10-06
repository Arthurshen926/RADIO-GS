"""Summarize completed mechanism controls without selecting benchmark winners."""
import json
from pathlib import Path


base = Path('output/region_native_next39')
result = {'development': True, 'scenes': ['figurines', 'ramen'], 'methods': {}}
for method in ['unit_mean', 'normalized_pool', 'retrieval_relations']:
    records = []
    for scene in result['scenes']:
        out = base / method / ('lerf_'+scene)
        evaluation = json.loads((out/'anchored/pipeline_calibrated/evaluation.json').read_text())
        image = json.loads((out/'reference_image_evaluation.json').read_text())
        fit = json.loads((out/'fit.json').read_text())
        metrics = {k: {**{m: v[m] for m in ['iou', 'ap', 'localization']}, 'selected_iou': v['selected']['miou']} for k, v in evaluation['metrics'].items()}
        records.append(dict(scene=scene, text=metrics, image=image['macro'], fit=fit))
    macro = {name: {metric: sum(r['text'][name][metric] for r in records)/len(records) for metric in records[0]['text'][name]} for name in records[0]['text']}
    image_macro = {name: {metric: sum(r['image'][name][metric] for r in records)/len(records) for metric in records[0]['image'][name]} for name in records[0]['image']}
    result['methods'][method] = dict(records=records, macro=macro, image_macro=image_macro)
(base/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: {'text':v['macro'], 'image':v['image_macro']} for k,v in result['methods'].items()},indent=2))
