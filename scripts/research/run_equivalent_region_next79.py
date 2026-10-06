"""Native-resolution evidence writing with fixed semantic calibration and extent."""
import argparse,json,os,subprocess,shutil
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();key='lerf_'+a.scene;root=Path('output/equivalent_region_next79');base=root/'all';env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python';out=base/'semantic_only'/key;out.mkdir(parents=True,exist_ok=True)
commands=[[py,'scripts/research/rebuild_expanded_semantic_next72.py','--key',key,'--cohort','all','--fit-width','256','--output-base',str(root),'--region-descriptors',str(Path('output/region_relations_next79')/a.scene/'descriptors.pt')]]
for name in ['refine','align']:commands.append([py,f'scripts/research/{name}_joint_semantic_next66.py','--key',key,'--arm','all','--recipe-root',str(root)])
rows=[]
with (out/'execution.log').open('w') as log:
 for cmd in commands:
  rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(command=cmd,exit_code=rc));(out/'execution.json').write_text(json.dumps(rows,indent=2))
  if rc:raise RuntimeError((key,rc,cmd[1]))
 source=base/'calibrated'/key;old=Path('output/native_write_next75/all/joint_extent')/key
 payload=torch.load(source/'typed_readout.pt',map_location='cpu',weights_only=False);selection=torch.load(old/'typed_readout.pt',map_location='cpu',weights_only=False)
 for name in ['selection_scores','selection_geometry','selection_observed']:payload[name]=selection[name]
 torch.save(payload,out/'typed_readout.pt')
 for name in ['field.pt','calibration.pt']:shutil.copyfile(source/name,out/name)
 cmd=[py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(base/'semantic_only'),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')]
 rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(command=cmd,exit_code=rc));(out/'execution.json').write_text(json.dumps(rows,indent=2))
 if rc:raise RuntimeError((key,rc,cmd[1]))
