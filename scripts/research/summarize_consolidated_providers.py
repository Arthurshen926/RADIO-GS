"""Compare the two predeclared provider/calibration controls across all scenes."""
import json
from pathlib import Path

root=Path('output/research_consolidation')
result={}
for arm in ['source_run','regional_run','calibrated_run']:
    rows=[]
    for scene in ['figurines','ramen','teatime','waldo_kitchen']:
        path=root/arm/scene/'evaluation/evaluation.json'
        metrics=json.loads(path.read_text())['metrics']['hierarchical_pipeline']
        rows.append(dict(scene=scene,report=str(path),semantic_iou=metrics['iou'],
                         ap=metrics['ap'],selected_projection_iou=metrics['selected']['miou']))
    result[arm]=dict(scenes=rows,macro={k:sum(r[k] for r in rows)/len(rows)
                    for k in ['semantic_iou','ap','selected_projection_iou']})
(root/'provider_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
for arm,data in result.items():print(arm,data['macro'])
