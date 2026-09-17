"""Bind experimental corrected geometry to fresh source-only contributions."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
from radio_gs.scripts.build_frozen_geometry_authority import build
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args()
base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=base/'local_calibrated_geometry_20260916'/a.scene
r=json.loads((root/'report.json').read_text());original=json.loads((base/'geometry_only_repair_final_20260914'/a.scene/'report.json').read_text())
report=dict(source_report=original['source_report'],source_report_sha256=original['source_report_sha256'],mapping_frames=r['mapping_frames'],heldout_frames=r['heldout_frames'],results={'independent_scale':{'geometry':r['candidate_geometry'],'geometry_sha256':r['candidate_geometry_sha256']}},experimental_only=True,not_promoted=True,local_calibration_report_sha256=sha256_file(root/'report.json'),local_calibration_fields_sha256=sha256_file(root/'local_calibration.pt'))
pth=root/'authority_geometry_report.json'
if pth.exists():raise FileExistsError(pth)
pth.write_text(json.dumps(report,indent=2)+'\n');info=json.loads((base/'preserved_source_inventory_20260916'/a.scene/'teachers.json').read_text())
print(build(SimpleNamespace(geometry_report=str(pth),expected_report_sha256=sha256_file(pth),frame_ids=','.join(map(str,info['train_frames'])),height=96,width=128,device=a.device,output_dir=str(root/'native_diagnostic_authority'))))
