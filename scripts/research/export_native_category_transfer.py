"""Make the matched native-map/category-head recipe an explicit public input."""
import json
from pathlib import Path
import torch
from radio_gs.querying.native_category_head import export_head
root=Path('output/native_category_transfer');configs=Path('configs/pipeline/scannet_native_calibrated');configs.mkdir(parents=True,exist_ok=True)
rows=json.loads(Path('configs/consolidated/scannet/cohort.json').read_text())['scenes'];cohort=[]
for row in rows:
 name=row['scene'];old=json.loads(Path(row['scene_config']).read_text());scene=json.loads(Path(f'output/scannet_native_completion/{name}/scene.json').read_text())
 head=root/name/'category_head.pt';torch.save(export_head(old),head)
 scene.update(native_category_head=str(head),source_evidence=old['source_evidence'],calibration=dict(kind='explicit_source_trained_native_category_head',parameters=str(head)),category_evidence_frames=[v['frame'] for v in old['source_evidence']['views']])
 scene['map_contract']['source_trained_category_calibration']=True
 path=configs/(name+'.scene.json');path.write_text(json.dumps(scene,indent=2)+'\n');cohort.append(dict(row,scene_config=str(path)))
(configs/'cohort.json').write_text(json.dumps(dict(scenes=cohort,all_source_maps=True,category_evidence='same176 cached source views; not all1158',historical_predictions_as_inputs=False),indent=2)+'\n')
