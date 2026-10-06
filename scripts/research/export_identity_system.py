"""Combine validated region information and detector-relative view confidence.

The per-query known-pixel confidence budget remains the native-budget baseline;
there is no scene-specific override or parameter sweep.
"""
import json
from pathlib import Path
root=Path('configs/pipeline/scannet_identity_observations');root.mkdir(parents=True,exist_ok=True)
recipe=json.loads(Path('configs/pipeline/system_v4.json').read_text())
recipe['name']='RADIO-GS native observation system v5'
recipe['role']='information-weighted native observations and detector-relative source confidence with preserved per-query evidence budget'
for job in recipe['jobs']:
    if job['benchmark']!='scannet':continue
    scene=json.loads(Path(job['scene']).read_text())
    scene['source_identity_control']='detector_matched_mass'
    destination=root/(job['name']+'.scene.json');destination.write_text(json.dumps(scene,indent=2)+'\n')
    job['scene']=str(destination)
recipe['comparison']='v3 baseline and v4 entropy-only control retained; uniform rule for all ten scenes'
Path('configs/pipeline/system_v5.json').write_text(json.dumps(recipe,indent=2)+'\n')
