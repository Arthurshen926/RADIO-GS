"""Use every eligible source32 mapping view; preserve validation and geometry."""
import argparse,json,os,shutil,subprocess
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
old=Path('output/depth_opacity_pipeline_20260918')/a.scene/'base';root=Path('output/full_field_rebuild_next19')/a.scene;base=root/'base';base.mkdir(parents=True,exist_ok=False)
plan=json.loads((old/'source_plan.json').read_text());ap=Path(plan['source_authority']['path']);assert sha256_file(ap)==plan['source_authority']['sha256'];authority=json.loads(ap.read_text());policy=authority['information_policy'];assert not policy['benchmark_ground_truth_used'] and not policy['query_text_used'];gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json';geo=json.loads(gp.read_text());rp=Path(plan['radio_root']);rm=rp/'frame_manifest.json';radio=json.loads(rm.read_text());rb={r['frame_idx']:r for r in radio['frames']};images={int(r['image_id'].split('_')[-1]):r for r in authority['images']};eligible=sorted(set(images)&set(rb)&set(geo['mapping_frames']));validation=plan['validation_frames'];train=sorted(set(eligible)-set(validation));assert set(plan['train_frames'])<=set(train);assert set(validation)<=set(eligible);assert not set(train)&set(validation)
for f in train+validation:
 rec=images[f];assert sha256_file(rec['path'])==rec['sha256'];assert rb[f]['source_sha256']==rec['sha256'] and rb[f]['source_file']==Path(rec['path']).name
new=dict(plan,frames=train+validation,train_frames=train,validation_frames=validation,original_train_frames=plan['train_frames'],eligible_frames=eligible,added_train_frames=sorted(set(train)-set(plan['train_frames'])),geometry_report_sha256=sha256_file(gp),radio_manifest_sha256=sha256_file(rm),selection='All predeclared source32 views with frozen geometry mapping permission and native source assets; original validation excluded. No GT or query used.',benchmark_data_used=False,expanded_input_budget=True,full_field_reconstruction=True,geometry_rebuilt=False,implementation_sha256=sha256_file(__file__))
(base/'source_plan.json').write_text(json.dumps(new,indent=2)+'\n')
print(a.scene,'train',len(train),'validation',validation,'added',new['added_train_frames'],flush=True)
if a.plan_only:raise SystemExit(0)
sam=base/'source_sam';sam.mkdir();manifest=json.loads((old/'source_sam/manifest.json').read_text());assert manifest['source_authority']==plan['source_authority'];reuse={}
for rec in manifest['images']:
 f=int(rec['image_id'].split('_')[-1])
 if f in new['frames'] and manifest['runtime_binding']['cuda']['requested_device']=='cuda:0':
  source=Path(rec['output']);assert sha256_file(source)==rec['output_sha256'];shutil.copy2(source,sam/source.name);reuse[source.name]=rec['output_sha256']
execution=json.loads((old/'source_sam_execution.json').read_text());cmd=execution['command'].copy();cmd[cmd.index('--output-root')+1]=str(sam);cmd[cmd.index('--image-ids')+1]=','.join(f'frame_{f:05d}' for f in new['frames']);cmd[cmd.index('--device')+1]='cuda:0'
if '--skip-existing' not in cmd:cmd.append('--skip-existing')
snap=base/'sam_sources';snap.mkdir();impl={}
for f in ['radio_gs/scripts/build_sam3_multiscale_hierarchy_cache.py','radio_gs/scripts/build_sam3_foundation_cache.py','radio_gs/models/sam3_multiscale_hierarchy.py','scripts/audits/run_lerf_source_stage.py',__file__]:
 shutil.copy2(f,snap/Path(f).name);impl[f]=sha256_file(f)
receipt={'command':cmd,'implementations':impl,'status':'running','source_plan_sha256':sha256_file(base/'source_plan.json'),'reused_2d_payloads':reuse,'no_geometry_indexed_assets_reused':True,'sam_reuse_requires_same_logical_device':True};marker=base/'source_sam_execution.json';marker.write_text(json.dumps(receipt,indent=2)+'\n')
with (base/'source_sam.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
receipt.update(status='complete' if r.returncode==0 else 'failed',returncode=r.returncode)
if r.returncode==0:
 newmanifest=json.loads((sam/'manifest.json').read_text());assert newmanifest['generation_contract']==manifest['generation_contract'];receipt['manifest_sha256']=sha256_file(sam/'manifest.json')
marker.write_text(json.dumps(receipt,indent=2)+'\n');assert r.returncode==0
