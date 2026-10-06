"""Freeze one complete native system recipe, with one rule per task type."""
import json
from pathlib import Path

root = Path('configs/pipeline/scannet_native_complete')
root.mkdir(parents=True, exist_ok=True)
recipe = json.loads(Path('configs/pipeline/system_complete_sources.json').read_text())
recipe['name'] = 'RADIO-GS complete native system v2'
recipe['role'] = 'complete-source native field and typed task readout; no historical scene calibration in ScanNet'
for job in recipe['jobs']:
    if job['benchmark'] != 'scannet':
        continue
    scene = json.loads(Path(job['scene']).read_text())
    scene['source_category_fusion'] = 'visibility_weighted_vote'
    destination = root/(job['name']+'.scene.json')
    destination.write_text(json.dumps(scene, indent=2)+'\n')
    job['scene'] = str(destination)
recipe['claims'] = dict(compact_learned_multiteacher=False, independent_3d_lerf_ground_truth=False, annotated_four_query_accuracy=False, official_sota_equivalence=False)
Path('configs/pipeline/system_v2.json').write_text(json.dumps(recipe, indent=2)+'\n')
