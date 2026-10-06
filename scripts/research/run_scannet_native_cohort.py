"""All ten ScanNet maps, identical all-source native recipe and dynamic GPU lanes.

Reusable SAM outputs are matched by source RGB hash and the same generation
recipe. Missing frames are generated in balanced batches; no view-count cap.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
SAM='/root/miniconda3/envs/sam3/bin/python'
ROOT=Path('output/scannet_native_completion')
SOURCE=Path('output/scannet_public_next72')
ENCODER='a713301b217d38485fb2204c808367d10bc3cc40'

def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def prepare():
 rows=[]
 old_roots=[Path('output/scannet_expanded_budget_next29'),Path('output/scannet_coverage_native_next28'),Path('output/scannet_coverage_native_probe_20260919'),Path('output/scannet_frozen_field_20260918')]
 for source in sorted(SOURCE.glob('scene*/source_rgb.json')):
  scene=source.parent.name;images=json.loads(source.read_text())['images'];dest=ROOT/scene/'source_sam';dest.mkdir(parents=True,exist_ok=True);reused=[]
  for image in images:
   target=dest/(image['image_id']+'.pt')
   if target.exists():
    if target.is_symlink():reused.append(dict(frame=image['image_id'],path=str(target.resolve())))
    continue
   for old in old_roots:
    candidate=old/scene/'base/source_sam'/target.name
    if not candidate.exists():continue
    d=torch.load(candidate,map_location='cpu',weights_only=False);meta=d['metadata'];contract=meta['generation_contract']
    if meta['source_image']['sha256']==image['sha256'] and contract['digest']=='8d2aa5d01d70ca8c7aeb864fa28341f2dd6a18059cfdec7157777909c20e60f0':
     target.symlink_to(candidate.resolve());reused.append(dict(frame=image['image_id'],path=str(candidate)));break
  missing=[im['image_id'] for im in images if not (dest/(im['image_id']+'.pt')).exists()]
  rows.append(dict(scene=scene,source_authority=str(source),total_frames=len(images),missing_frames=missing,reused=reused))
 save(ROOT/'source_plan.json',dict(scenes=rows,total_source_frames=sum(r['total_frames'] for r in rows),new_teacher_frames=sum(len(r['missing_frames']) for r in rows),no_source_cap=True,same_generation_recipe=True))
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpus-file',default='output/scannet_native_completion/gpus.json');p.add_argument('--batch-frames',type=int,default=24);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();torch.set_num_threads(2);start=time.monotonic();rows=prepare()
 if a.prepare_only:return
 # Small scenes finish early; batches from large scenes distribute across lanes.
 pending=[]
 for row in sorted(rows,key=lambda r:r['total_frames']):
  for i in range(0,len(row['missing_frames']),a.batch_frames):pending.append(dict(stage='sam',scene=row['scene'],frames=row['missing_frames'][i:i+a.batch_frames],batch=i//a.batch_frames))
 active={};done=[];built=set();pool=ThreadPoolExecutor(6)
 def run(job,gpu):
  scene=job['scene'];out=ROOT/scene;env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4');commands=[];t=time.monotonic()
  if job['stage']=='sam':
   authority=SOURCE/scene/'source_rgb.json';commands=[[SAM,'-m','radio_gs.scripts.build_sam3_multiscale_hierarchy_cache','--source-authority',str(authority),'--source-authority-sha256',hashlib.sha256(authority.read_bytes()).hexdigest(),'--image-ids',','.join(job['frames']),'--output-root',str(out/'source_sam'),'--manifest-name',f"batch_{job['batch']:03d}.json",'--checkpoint-path','checkpoints/sam3_modelscope/sam3.pt','--checkpoint-sha256','9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e','--device','cuda:0']]
  else:
   cfg=Path('configs/consolidated/scannet_native');cfg.mkdir(parents=True,exist_ok=True)
   from radio_gs.scannet_constants import NYU40_ID_TO_NAME,OPENGAUSSIAN_NYU40_CLASS_SPLITS as splits
   ids=sorted(set(sum(splits.values(),[])));q=dict(queries=[dict(id=str(c),modality='text',intent='category',text=NYU40_ID_TO_NAME[c],class_id=c) for c in ids]);save(cfg/(scene+'.queries.json'),q)
   save(cfg/(scene+'.evaluation.json'),dict(scene=scene,ground_truth=str(Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/(scene+'_vh_clean_2.labels.ply')),splits=splits,official_protocol_equivalence_verified=False,independent_test=False))
   commands=[[PY,'scripts/research/build_scannet_native_map.py','--scene',scene],[PY,'scripts/research/run_scannet_query.py','--scene',str(out/'scene.json'),'--query',str(cfg/(scene+'.queries.json')),'--output',str(out/'query'),'--gpu',str(gpu),'--source-evidence','none'],[PY,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'query/typed_readout.pt'),'--evaluation',str(cfg/(scene+'.evaluation.json')),'--output',str(out/'evaluation'),'--point-only']]
  label=job['stage']+(f"_{job['batch']:03d}" if job['stage']=='sam' else '')
  save(out/(label+'_commands.json'),commands)
  with (out/(label+'.log')).open('w') as log:
   for command in commands:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  return dict(job,gpu=gpu,seconds=time.monotonic()-t,status='complete')
 while pending or active or len(built)<len(rows):
  for gpu,(future,job) in list(active.items()):
   if not future.done():continue
   result=future.result();done.append(result);del active[gpu];print(job['scene'],job['stage'],'complete',round(result['seconds'],1),flush=True)
   if job['stage']=='map':built.add(job['scene'])
  queued_scenes={j['scene'] for j in pending if j['stage']=='sam'}|{j['scene'] for _,j in active.values() if j['stage']=='sam'}
  map_scenes={j['scene'] for j in pending if j['stage']=='map'}|{j['scene'] for _,j in active.values() if j['stage']=='map'}|built
  for row in rows:
   if row['scene'] not in queued_scenes and row['scene'] not in map_scenes:
    pending.insert(0,dict(stage='map',scene=row['scene']))
  gpus=json.loads(Path(a.gpus_file).read_text())['gpus']
  for gpu in gpus:
   if gpu not in active and pending:
    job=pending.pop(0);active[gpu]=(pool.submit(run,job,gpu),job)
  save(ROOT/'execution.json',dict(elapsed_seconds=time.monotonic()-start,completed=done,active=[dict(j,gpu=g) for g,(_,j) in active.items()],pending=pending,gpus=gpus,complete=len(built)==len(rows),summed_worker_seconds=sum(x['seconds'] for x in done)))
  if active:time.sleep(2)
 pool.shutdown()
 reports={r['scene']:json.loads((ROOT/r['scene']/'evaluation/report.json').read_text()) for r in rows}
 summary=dict(scenes=reports,scene_macro={s:sum(v['point_metrics']['map'][s]['miou'] for v in reports.values())/len(reports) for s in ['19','15','10']},total_source_frames=sum(r['total_frames'] for r in rows),scenes_count=len(rows),recipe='full_source_sam3_three_layer_native_siglip1536_alpha_adjoint_unlabelled_mesh',historical_predictions_read=False,official_protocol_equivalence_verified=False,independent_test=False,worker_seconds=sum(x['seconds'] for x in done),wall_seconds=time.monotonic()-start)
 save(ROOT/'summary.json',summary);print(summary['scene_macro'],flush=True)
if __name__=='__main__':main()
