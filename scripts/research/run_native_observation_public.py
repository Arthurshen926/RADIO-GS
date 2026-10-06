"""Fresh public-entry reproduction of the complete native region response arm."""
import concurrent.futures,json,os,subprocess
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/native_observation_public');CONFIG=Path('configs/pipeline/native_observation')
CONFIG.mkdir(parents=True,exist_ok=True)
scenes=sorted(Path('configs/pipeline/scannet_native_complete').glob('*.scene.json'))
for path in scenes:
 scene=json.loads(path.read_text());scene['native_category_prior']['kind']='region_response_mixture'
 scene['map_contract']['extra_native_observation_assets']=True
 (CONFIG/path.name).write_text(json.dumps(scene,indent=2)+'\n')
def worker(gpu):
 for path in scenes[gpu::6]:
  name=path.name.removesuffix('.scene.json');out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
  with (out/'run.log').open('w') as log:
   subprocess.run([PYTHON,'-m','radio_gs.pipeline','--scene',str(CONFIG/path.name),'--queries',f'configs/consolidated/scannet_native/{name}.queries.json','--access','source-assisted','--output',str(out/'query'),'--gpu',str(gpu)],env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,check=True)
   subprocess.run([PYTHON,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'query/category/typed_readout.pt'),'--evaluation',f'configs/consolidated/scannet_native/{name}.evaluation.json','--output',str(out/'evaluation'),'--point-only'],env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,check=True)
  print(name,'public complete',flush=True)
with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(worker,range(6)))
rows={p.name.removesuffix('.scene.json'):json.loads((ROOT/p.name.removesuffix('.scene.json')/'evaluation/report.json').read_text())['point_metrics'] for p in scenes}
macro={a:{s:sum(r[a][s]['miou'] for r in rows.values())/len(rows) for s in ['19','15','10']} for a in ['map','source']}
reference=json.loads(Path('output/native_pooling_order/summary.json').read_text())['scenes']
difference=max(abs(rows[n][a][s]['miou']-reference[n]['region'][a][s]['miou']) for n in rows for a in ['map','source'] for s in ['19','15','10'])
(ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,maximum_metric_difference=difference,independent_test=False),indent=2)+'\n');print(macro,difference,flush=True)
