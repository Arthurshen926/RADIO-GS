"""One globally fixed partial-category observation candidate on all ten scenes."""
import concurrent.futures,json,os,subprocess
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/categorical_observation');CONFIG=Path('configs/pipeline/categorical_observation')
CONFIG.mkdir(parents=True,exist_ok=True)
scenes=sorted(Path('configs/pipeline/scannet_native_complete').glob('*.scene.json'))
for source in scenes:
 d=json.loads(source.read_text());d['source_category_fusion']='categorical_observation_vote';(CONFIG/source.name).write_text(json.dumps(d,indent=2)+'\n')
def worker(gpu):
 for source in scenes[gpu::6]:
  name=source.name.removesuffix('.scene.json');out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
  with (out/'run.log').open('w') as log:
   subprocess.run([PYTHON,'-m','radio_gs.pipeline','--scene',str(CONFIG/source.name),'--queries',f'configs/consolidated/scannet_native/{name}.queries.json','--access','source-assisted','--output',str(out/'query'),'--gpu',str(gpu)],env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,check=True)
   subprocess.run([PYTHON,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'query/category/typed_readout.pt'),'--evaluation',f'configs/consolidated/scannet_native/{name}.evaluation.json','--output',str(out/'evaluation'),'--point-only'],env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,check=True)
  print(name,'complete',flush=True)
with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(worker,range(6)))
rows={p.name.removesuffix('.scene.json'):json.loads((ROOT/p.name.removesuffix('.scene.json')/'evaluation/report.json').read_text())['point_metrics']['source'] for p in scenes}
macro={s:sum(r[s]['miou'] for r in rows.values())/len(rows) for s in ['19','15','10']}
(ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,independent_test=False,complete_vocabulary=True),indent=2)+'\n');print(macro,flush=True)
