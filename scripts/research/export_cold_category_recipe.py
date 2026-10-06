"""Export a cold-start alternative without overriding the retained recipe."""
import json
from pathlib import Path
folder=Path('configs/pipeline/scannet_cold_category');folder.mkdir(parents=True,exist_ok=True)
rows=json.loads(Path('configs/pipeline/scannet_native_calibrated/cohort.json').read_text())['scenes'];new=[]
for r in rows:
 scene=json.loads(Path(r['scene_config']).read_text());scene.pop('native_category_head')
 scene['native_category_prior']=dict(kind='mutually_exclusive_native_log_odds',temperature=100.,vocabulary='complete declared ScanNet category query list',historical_scene_parameters=False)
 scene['calibration']=dict(kind='native_categorical_prior_not_empirically_calibrated')
 scene['map_contract']['source_trained_category_calibration']=False
 dest=folder/(r['scene']+'.scene.json');dest.write_text(json.dumps(scene,indent=2)+'\n');new.append(dict(r,scene_config=str(dest)))
(folder/'cohort.json').write_text(json.dumps(dict(scenes=new,role='cold-start alternative; not global promotion'),indent=2)+'\n')
recipe=json.loads(Path('configs/pipeline/system_v1.json').read_text());recipe['name']='RADIO-GS cold-category alternative';recipe['role']='cold-start ScanNet prior alternative; retained v1 remains default'
paths={r['scene']:r['scene_config'] for r in new}
for j in recipe['jobs']:
 if j['benchmark']=='scannet':j['scene']=paths[j['name']]
Path('configs/pipeline/system_cold_category.json').write_text(json.dumps(recipe,indent=2)+'\n')
