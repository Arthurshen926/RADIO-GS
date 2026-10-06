"""Fixed ten-scene positive-region experiment; evaluation follows predictions."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['region','pixel_control'],default='region');args=p.parse_args()
NAME='region_positive_grounding' if args.mode=='region' else 'pixel_positive_grounding'
ROOT=Path('output')/NAME;CONFIG=Path('configs/pipeline')/NAME
SCENES=sorted(Path('configs/pipeline/scannet_identity_observations').glob('*.scene.json'))
GPUS=['0','1','2','3','4','5']
CONFIG.mkdir(parents=True,exist_ok=True)
for path in SCENES:
 scene=json.loads(path.read_text());scene['source_region_grounding']=args.mode
 scene['source_region_directory']=f"output/scannet_native_completion/{path.name.removesuffix('.scene.json')}/native_regions"
 (CONFIG/path.name).write_text(json.dumps(scene,indent=2)+'\n')
def worker(i):
 for path in SCENES[i::len(GPUS)]:
  name=path.name.removesuffix('.scene.json');out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
  env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=GPUS[i])
  with (out/'run.log').open('w') as log:
   subprocess.run([PYTHON,'scripts/research/run_scannet_query.py','--scene',str(CONFIG/path.name),'--query',f'configs/consolidated/scannet_native/{name}.queries.json','--output',str(out/'query'),'--gpu',GPUS[i]],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   subprocess.run([PYTHON,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'query/typed_readout.pt'),'--evaluation',f'configs/consolidated/scannet_native/{name}.evaluation.json','--output',str(out/'evaluation'),'--point-only'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   # Intermediate is explicitly source-assisted, never relabelled as map-only.
   subprocess.run([PYTHON,'-c',"import torch,sys; x=torch.load(sys.argv[1]); x['category_logits']=x['source_grounded_region_logits']; torch.save(x,sys.argv[2])",str(out/'query/typed_readout.pt'),str(out/'query/grounded_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   subprocess.run([PYTHON,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'query/grounded_readout.pt'),'--evaluation',f'configs/consolidated/scannet_native/{name}.evaluation.json','--output',str(out/'grounded_evaluation'),'--point-only'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  print(name,'complete',flush=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(len(GPUS)) as pool:list(pool.map(worker,range(len(GPUS))))
 rows={p.name.removesuffix('.scene.json'):{arm:json.loads((ROOT/p.name.removesuffix('.scene.json')/directory/'report.json').read_text())['point_metrics'] for arm,directory in [('final','evaluation'),('grounded','grounded_evaluation')]} for p in SCENES}
 macro={arm:{s:sum(r[arm]['source'][s]['miou'] for r in rows.values())/len(rows) for s in ['19','15','10']} for arm in ['final','grounded']}
 (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,new_teacher_calls=0,independent_test=False,grounded_access='source-assisted',baseline='system_v5',rules_fixed_before_evaluation=True),indent=2)+'\n');print(macro,flush=True)
