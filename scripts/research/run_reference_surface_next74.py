"""Execute source-reference correspondence and frozen-map query reconstruction."""
import argparse,subprocess,os,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scenes',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH='.',OMP_NUM_THREADS='4');py='/root/miniconda3/envs/cybersim_agent/bin/python';sam='/root/miniconda3/envs/sam3/bin/python';base=Path('output/reference_surface_next74')
for scene in a.scenes.split(','):
 folder=base/scene;folder.mkdir(parents=True,exist_ok=True);commands=[]
 if not (folder/'anchors.json').exists():commands.append([py,'scripts/research/prepare_reference_anchors_next74.py','--scene',scene])
 if not (folder/'decoding.json').exists():commands.append([sam,'scripts/research/decode_reference_anchors_next74.py','--scene',scene,'--base',str(base)])
 for mode in ['replace','union']:
  if not (base/('eval_'+mode)/'new'/scene/'report.json').exists():commands.append([py,'scripts/research/fit_nvos_geometry_transfer_next72.py','--scene',scene,'--carrier','new','--tracks-base',str(base/mode),'--output-base',str(base/('eval_'+mode))])
 rows=[]
 with (folder/'execution.log').open('w') as log:
  for cmd in commands:
   rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(command=cmd,exit_code=rc));(folder/'execution.json').write_text(json.dumps(rows,indent=2))
   if rc:raise RuntimeError((scene,rc,cmd[1]))
