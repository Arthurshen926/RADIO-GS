"""Export one uniform observation-field recipe; retain v2 as a comparator."""
import json
from pathlib import Path

root=Path('configs/pipeline/scannet_observation_complete');root.mkdir(parents=True,exist_ok=True)
recipe=json.loads(Path('configs/pipeline/system_v2.json').read_text())
recipe['name']='RADIO-GS native observation system v3'
recipe['role']='native-coordinate fields and typed readouts; ScanNet uses a query-independent region dictionary and sparse element weights'
for job in recipe['jobs']:
    if job['benchmark']!='scannet':continue
    scene=json.loads(Path('output/compiled_native_observations',job['name'],'scene.json').read_text())
    for key in ['native_field','native_region_directories','source_cameras']:scene.pop(key,None)
    destination=root/(job['name']+'.scene.json');destination.write_text(json.dumps(scene,indent=2)+'\n')
    job['scene']=str(destination)
recipe['claims']['observation_compilation_is_new_accuracy_gain']=False
recipe['claims']['compiled_category_field_is_four_query_validation']=False
recipe['comparison']='system_v2 retained; one globally fixed category rule, no per-scene choice'
Path('configs/pipeline/system_v3.json').write_text(json.dumps(recipe,indent=2)+'\n')
