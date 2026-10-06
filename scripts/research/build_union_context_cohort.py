import concurrent.futures,json,os,subprocess
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python';ROOT=Path('output/category_union_context');GPUS=['0','1','2','3','4','5']
SCENES=sorted(Path('configs/pipeline/scannet_identity_observations').glob('*.scene.json'))
def worker(i):
 for p in SCENES[i::len(GPUS)]:
  name=p.name.removesuffix('.scene.json');out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
  with (out/'build.log').open('w') as f:
   subprocess.run([PYTHON,'-m','radio_gs.pipeline.category_union_identity','--scene',str(p),'--output',str(out),'--representation','context'],env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=GPUS[i]),stdout=f,stderr=subprocess.STDOUT,check=True)
  print(name,'encoded',flush=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(worker,range(6)))
 rows={p.name.removesuffix('.scene.json'):json.loads((ROOT/p.name.removesuffix('.scene.json')/'encoding.json').read_text()) for p in SCENES}
 (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,new_native_image_encodings=sum(r['new_native_image_encodings'] for r in rows.values()),new_sam_calls=0),indent=2)+'\n')
