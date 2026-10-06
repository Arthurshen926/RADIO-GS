"""Paired full-cohort comparison of source region representations and readouts."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
parser=argparse.ArgumentParser()
parser.add_argument('--evidence',default='output/category_union_context')
parser.add_argument('--representations',nargs='+',default=['natural_global','context_region'])
parser.add_argument('--name',default='union_context_control')
args=parser.parse_args()
ROOT=Path('output')/args.name;CONFIG=Path('configs/pipeline')/args.name
MODES=[f'{representation}__{readout}' for readout in ['categories','conflict'] for representation in args.representations]
SCENES=sorted(Path('configs/pipeline/scannet_identity_observations').glob('*.scene.json'))
GPUS=['0','1','2','3','4','5'];JOBS=[]
for mode in MODES:
 (CONFIG/mode).mkdir(parents=True,exist_ok=True)
 for path in SCENES:
  scene=json.loads(path.read_text());representation,readout=mode.split('__');scene['source_union_identity']=str(Path(args.evidence)/path.name.removesuffix('.scene.json')/representation)
  if readout=='categories':scene['source_category_fusion']='union_native_categories'
  else:scene['source_category_ambiguity']='union_conflict'
  dest=CONFIG/mode/path.name;dest.write_text(json.dumps(scene,indent=2)+'\n');JOBS.append((mode,dest))
def worker(i):
 for mode,path in JOBS[i::len(GPUS)]:
  name=path.name.removesuffix('.scene.json');out=ROOT/mode/name;out.mkdir(parents=True,exist_ok=True)
  env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=GPUS[i])
  with (out/'run.log').open('w') as log:
   subprocess.run([PYTHON,'scripts/research/run_scannet_query.py','--scene',str(path),'--query',f'configs/consolidated/scannet_native/{name}.queries.json','--output',str(out/'query'),'--gpu',GPUS[i]],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   subprocess.run([PYTHON,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'query/typed_readout.pt'),'--evaluation',f'configs/consolidated/scannet_native/{name}.evaluation.json','--output',str(out/'evaluation'),'--point-only'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  print(mode,name,'complete',flush=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(len(GPUS)) as pool:list(pool.map(worker,range(len(GPUS))))
 rows={mode:{p.name.removesuffix('.scene.json'):json.loads((ROOT/mode/p.name.removesuffix('.scene.json')/'evaluation/report.json').read_text())['point_metrics'] for p in SCENES} for mode in MODES}
 macro={mode:{s:sum(r['source'][s]['miou'] for r in scenes.values())/len(scenes) for s in ['19','15','10']} for mode,scenes in rows.items()}
 (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,additional_identity_encoding=str(Path(args.evidence)/'summary.json'),new_sam_calls=0,independent_test=False,baseline='system_v5',rules_fixed_before_evaluation=True,geometry_and_prior_unchanged=True),indent=2)+'\n');print(macro,flush=True)
