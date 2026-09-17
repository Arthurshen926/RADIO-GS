"""Rebuild episode-dependent assets and run a complete matched-budget field fit."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args()
 base=Path('/mnt/pool/sqy/results/RADIO-GS/output');old=base/'preserved_source_inventory_20260916'/a.scene;root=base/'scope_consistent_episodes_20260916'/a.scene
 marker=root/'scope_reconstruction_execution_20260917.json'
 if marker.exists():raise FileExistsError(marker)
 for name in ['source_frames.json','native_query_encoder.json','source_grounded_text.pt','source_grounded_text.json']:
  if (root/name).exists():raise FileExistsError(root/name)
  shutil.copy2(old/name,root/name)
 jobs=json.loads(Path('paper/artifacts/preserved_inventory_training_protocol_20260916.json').read_text())['jobs'];job=next(j for j in jobs if j['scene']==a.scene)
 train=[x.replace(str(old),str(root)).replace('preserved96_protected','scope96_protected') for x in job['command']];train[train.index('--device')+1]=a.device
 commands=[
 [sys.executable,'scripts/audits/build_source_fusion_initialization.py','--root',str(root),'--output',str(root/'source_fusion_initialization.pt')],
 [sys.executable,'scripts/audits/compile_grounded_text_supervision.py','--root',str(root),'--grounding',str(root/'source_grounded_text.pt'),'--output',str(root/'grounded_text_supervision.pt'),'--device',a.device],train,
 [sys.executable,'scripts/audits/calibrate_frozen_source_images.py','--root',str(root),'--run','scope96_protected','--device',a.device]]
 record=dict(scene=a.scene,same_training_steps_as_parent=job['steps'],commands=commands,completed=[])
 marker.write_text(json.dumps(record,indent=2)+'\n')
 for index,cmd in enumerate(commands):
  print('starting',index,cmd,flush=True)
  with (root/f'scope_reconstruction_stage_{index}_20260917.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
  record['completed'].append(index);marker.write_text(json.dumps(record,indent=2)+'\n');print('completed',index,flush=True)

if __name__=='__main__':main()
