"""Frozen-geometry NVOS fern pipeline pilot; same protected base field recipe.

This does not claim the caption-augmented complete LERF method is reproduced.
"""
import json,os,subprocess,sys
from pathlib import Path
import torch
from radio_gs.training.reliable_correspondence import source_region_query_tasks
from radio_gs.utils.immutable_artifacts import sha256_file

root=Path('/mnt/pool/sqy/results/RADIO-GS/output/nvos_frozen_authority_20260916/fern')
info=json.loads((root/'teachers.json').read_text());ep=torch.load(root/'episodes.pt',weights_only=False);g=torch.load(root/'grounded_text_supervision.pt',weights_only=False)
indexed=[(i,e) for i,e in enumerate(ep['episodes']) if e['split']=='train' and e['modality']!='text']+[(len(ep['episodes'])+i,e) for i,e in enumerate(g['tasks'])]
indexed,_=source_region_query_tasks(indexed,ep,[info['frames'].index(f) for f in info['train_frames']]);counts={m:sum(e['modality']==m for _,e in indexed) for m in ('text','image','prompt')}
steps=6*max(counts.values())
cmd=[sys.executable,'radio_gs/scripts/train_three_query_field.py','--root',str(root),'--geometry','/mnt/pool/sqy/results/RADIO-GS/output/nvos_moge_bound_20260916/fern/frozen_geometry/independent_scale.pt','--name','nvos96_protected','--steps',str(steps),'--latent-dim','96','--native-semantic-dim','64','--source-region-query-tasks','--share-siglip-adapter','--calibrated-readout','--native-decoder-readout','--source-fusion-initialization',str(root/'source_fusion_initialization.pt'),'--freeze-native-decoder','--grounded-text-supervision',str(root/'grounded_text_supervision.pt'),'--protect-native-semantic','--teachers','dino,sam,siglip','--device','cuda:0']
commands=[cmd,[sys.executable,'scripts/audits/calibrate_frozen_source_images.py','--root',str(root),'--run','nvos96_protected','--device','cuda:0']]
record=dict(schema='nvos_frozen_field_execution.v1',scene='fern',source_only=True,benchmark_data_used=False,full_method=False,
 counts=counts,steps=steps,budget='two complete balanced modality coverage cycles, 6*max(task counts)',commands=commands,completed=[],
 source_bindings={name:sha256_file(root/name) for name in ['teachers.json','episodes.pt','grounded_text_supervision.pt','source_fusion_initialization.pt','native_query_encoder.json']})
p=root/'field_execution_20260917.json';assert not p.exists();p.write_text(json.dumps(record,indent=2)+'\n')
for i,command in enumerate(commands):
 print('starting',i,steps,counts,flush=True)
 with (root/f'field_stage_{i}_20260917.log').open('x') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
 record['completed'].append(i);p.write_text(json.dumps(record,indent=2)+'\n')
