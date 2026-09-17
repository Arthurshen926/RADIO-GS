"""Verify immutable parents and changed calibration keys for this control."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
base=Path('/mnt/pool/sqy/results/RADIO-GS/output/component_retrieval_pipeline_20260915')
summary={'schema':'frozen_source_calibration_iteration.v1','joint_default_promoted':False,'text_calibration_adopted':False,'image_calibration_status':'optional source-only correction; incomplete joint performance','tests_passed':3,'scenes':{}}
for scene,n in [('figurines',2106),('ramen',3240)]:
 root=base/scene;stem=f'fusion_init_bounded002_{n}';runs={}
 for modality,parent in [('image',stem),('text',stem+'_imagecal_balanced')]:
  original=torch.load(root/parent/'field.pt',weights_only=False)
  for weighting in ['count','balanced']:
   name=parent+'_'+modality+'cal_'+weighting;path=root/name/'field.pt';report=json.loads((root/name/'report.json').read_text());payload=torch.load(path,weights_only=False);assert sha256_file(path)==report['field_sha256'];changed=[k for k,v in original['state_dict'].items() if not torch.equal(v,payload['state_dict'][k])];assert set(changed)=={f'readout_log_scale.{modality}',f'readout_bias.{modality}'};assert payload['geometry_binding']==original['geometry_binding']
   runs[name]={'sha256':sha256_file(path),'parent_sha256':sha256_file(root/parent/'field.pt'),'changed_keys':changed,'report_sha256':sha256_file(root/name/'report.json')}
 results={}
 for suffix in ['image_calibration','text_calibration_text','text_calibration_reference']:
  p=Path(f'paper/artifacts/{scene}_frozen_source_{suffix}_20260916.json');d=json.loads(p.read_text());results[suffix]={'sha256':sha256_file(p),'path':str(p),'runs':{k:({m:r[m] for m in ['observation_miou','average_precision_macro']} if 'observation_miou' in r else r['macro']) for k,r in d['runs'].items()}}
 summary['scenes'][scene]={'runs':runs,'results':results}
path=Path('paper/artifacts/frozen_source_calibration_iteration_20260916.json');assert not path.exists();path.write_text(json.dumps(summary,indent=2)+'\n');print('8 candidates verified: only the two intended scalar parameters changed.')
