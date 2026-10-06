"""One fixed entropy-information rule; no scene-specific parameter selection."""
import json
from pathlib import Path

root=Path('configs/pipeline/scannet_reliable_observations');root.mkdir(parents=True,exist_ok=True)
recipe=json.loads(Path('configs/pipeline/system_v3.json').read_text())
recipe['name']='RADIO-GS reliable native observation system v4'
recipe['role']='compiled native observation field; categorical information-weighted fusion; typed task readouts'
for job in recipe['jobs']:
    if job['benchmark']!='scannet':continue
    scene=json.loads(Path(job['scene']).read_text())
    scene['native_observation_reliability']='entropy_information'
    destination=root/(job['name']+'.scene.json');destination.write_text(json.dumps(scene,indent=2)+'\n')
    job['scene']=str(destination)
recipe['comparison']='v3 retained; full-cohort fixed rule, no per-scene routing; original v3 field assets reused'
Path('configs/pipeline/system_v4.json').write_text(json.dumps(recipe,indent=2)+'\n')
