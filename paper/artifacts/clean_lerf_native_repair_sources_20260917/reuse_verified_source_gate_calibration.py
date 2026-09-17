"""Reuse only a source precision calibration with identical text evidence/geometry."""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
c=json.loads((a.calibration/'report.json').read_text());assert c['source_only'] and not c['benchmark_data_used'];assert sha256_file(a.calibration/'field.pt')==c['field_sha256']
assert sha256_file(a.root/'episodes.pt')==c['episode_sha256']
calibrated_groups=a.calibration.parent/'source_phrase_groups.json'
assert sha256_file(calibrated_groups)==c['groups_sha256']
old_groups=json.loads(calibrated_groups.read_text());new_groups=json.loads((a.root/'source_phrase_groups.json').read_text())
# The group-builder receipt also identifies its field, which differs by design.
# Every actual source group, evidence identity and grouping rule must be equal.
assert {k:v for k,v in old_groups.items() if k!='parent_field_sha256'}=={k:v for k,v in new_groups.items() if k!='parent_field_sha256'}
path=a.root/'exact_render_query_threshold96/field.pt';r=json.loads(path.with_name('report.json').read_text());assert sha256_file(path)==r['field_sha256']
payload=torch.load(path,weights_only=False,map_location='cpu');other=torch.load(a.calibration/'field.pt',weights_only=False,map_location='cpu')
assert payload['query_encoder']==other['query_encoder'] and payload['geometry_binding']==other['geometry_binding']
for key in ['grounded_transfer.prototypes','grounded_transfer.routing_center']:
 assert torch.equal(payload['state_dict'][key],other['state_dict'][key]),key
threshold=payload['state_dict']['grounded_transfer.acceptance'];assert threshold.tolist()==[row['old_threshold'] for row in c['checks']]
new=other['state_dict']['grounded_transfer.acceptance'];state=dict(payload['state_dict']);state['grounded_transfer.acceptance']=new.clone()
for key,value in payload['state_dict'].items():
 if key!='grounded_transfer.acceptance':assert torch.equal(value,state[key])
out=a.root/'source_precision_gate96';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py');torch.save(dict(payload,state_dict=state),out/'field.pt')
report=dict(c,field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=r['field_sha256'],groups_sha256=sha256_file(a.root/'source_phrase_groups.json'),calibration_report_sha256=sha256_file(a.calibration/'report.json'),calibration_path=str(a.calibration.resolve()),implementation_sha256=sha256_file(__file__),reuse_verified_identical_source_queries_groups_prototypes_and_geometry=True)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(out)
