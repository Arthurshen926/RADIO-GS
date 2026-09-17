"""Resumable bound NVOS recipe: exact geometry -> source teachers -> image field.

Shared image branch pilot, not a claim of complete LERF caption-head identity.
Each stage is immutable and resumes only from a successful recorded output hash.
"""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output-root',type=Path);p.add_argument('--scale-prior',type=Path);p.add_argument('--device',required=True);p.add_argument('--resume-verified-completed',action='store_true');p.add_argument('--retry-failed-stage');p.add_argument('--allow-empty-text-supervision',action='store_true');p.add_argument('--minimum-voxel-views',type=int,choices=[1,2],default=2);p.add_argument('--query-root',type=Path);p.add_argument('--signed-policy',choices=['fixed','adaptive'],default='fixed');p.add_argument('--voxel-metres',type=float,choices=[.015,.0075],default=.015);p.add_argument('--until',choices=['geometry','teachers','field','all'],default='all');a=p.parse_args();torch.set_num_threads(4)
 base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=a.output_root or base/'nvos_bound_pipeline_20260917'/a.scene;root.mkdir(parents=True,exist_ok=True)
 pred=base/('nvos_moge_bound_20260916' if a.scene=='fern' else 'nvos_moge_bound_20260917')/a.scene
 source=next(x for x in json.loads((base/'nvos_geometry_sources_bound_20260916/report.json').read_text())['scenes'] if x['scene']==a.scene)
 marker=root/'execution.json';sources={str(Path(__file__)):sha256_file(__file__),'scripts/audits/run_nvos_source_stage.py':sha256_file('scripts/audits/run_nvos_source_stage.py')}
 record=json.loads(marker.read_text()) if marker.exists() else dict(schema='bound_nvos_pipeline.v1',scene=a.scene,source_authority_sha256=source['source_authority_sha256'],geometry_parameters=dict(width=320,voxel_metres=a.voxel_metres,mapping_stride=1,minimum_voxel_views=a.minimum_voxel_views),source_raster_hw=[96,128],support_extension_raster_hw=[384,512],full_caption_method=False,stages={})
 assert record['source_authority_sha256']==sha256_file(source['source_authority'])
 assert record['geometry_parameters']['voxel_metres']==a.voxel_metres,'voxel configuration differs'
 assert record['geometry_parameters']['minimum_voxel_views']==a.minimum_voxel_views,'voxel support differs'
 py='/root/miniconda3/envs/cybersim_agent/bin/python';sam='/root/miniconda3/envs/sam3/bin/python'
 def stage(name,entry,args,outputs,*,envpy=py,module=False,guard=True):
  impl=entry.replace('.','/')+'.py' if module else entry
  command=[envpy]
  if guard:command+=['scripts/audits/run_nvos_source_stage.py','--scene',a.scene]+(['--module'] if module else [])+[entry]
  else:command+=(['-m',entry] if module else [entry])
  command+=list(map(str,args));identity={impl:sha256_file(impl),**sources}
  if name in ('source_sam','source_grounding'):
   for dependency in ['radio_gs/scripts/build_sam3_foundation_cache.py','/root/external/sam3/sam3/model_builder.py','/root/external/sam3/sam3/model/vitdet.py']:
    identity[dependency]=sha256_file(dependency)
  if name in record['stages']:
   old=record['stages'][name]
   if old['status']=='complete':
    assert old['command']==command,(name,'stage config changed')
    if old['implementations']!=identity:
     if not a.resume_verified_completed:raise RuntimeError(f'{name}: implementation changed; explicit verified resume required')
     for f,digest in old['implementations'].items():
      assert sha256_file(root/'sources'/name/Path(f).name)==digest,(name,'historical implementation snapshot changed')
    for f,digest in old['outputs'].items():assert sha256_file(f)==digest,(name,'output changed')
    return
   if old['status']!='failed' or a.retry_failed_stage!=name:raise RuntimeError(f'{name} is incomplete; inspect its log before recovery')
   if any(Path(f).exists() for f in outputs):raise RuntimeError(f'{name}: partial outputs need explicit inspection; cannot overwrite')
   history=root/'failed_attempts'/name/str(old['started']);history.mkdir(parents=True,exist_ok=False)
   (history/'stage.json').write_text(json.dumps(old,indent=2)+'\n')
   shutil.move(str(root/(name+'.log')),str(history/'stage.log'))
   shutil.move(str(root/'sources'/name),str(history/'sources'))
   record.setdefault('failed_attempts',[]).append(str(history/'stage.json'))
  st=dict(command=command,implementations=identity,status='running',started=time.time());record['stages'][name]=st;marker.write_text(json.dumps(record,indent=2)+'\n')
  snap=root/'sources'/name;snap.mkdir(parents=True)
  for f in identity:
   dest=snap/Path(f).name;shutil.copy2(f,dest)
  print(a.scene,name,flush=True)
  with (root/(name+'.log')).open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'))
  st['returncode']=r.returncode;st['elapsed_seconds']=time.time()-st['started'];st['status']='failed' if r.returncode else 'complete'
  if not r.returncode:st['outputs']={str(f):sha256_file(f) for f in outputs}
  marker.write_text(json.dumps(record,indent=2)+'\n')
  if r.returncode:raise RuntimeError(f'{name} failed; see {root/(name+".log")}')
 if a.scale_prior is not None:
  cal=a.scale_prior
  cp=json.loads(cal.read_text());assert cp['source_only'] and not cp['benchmark_data_used']
  assert next(r['sha256'] for r in cp['geometry_receipt']['inputs'] if r['role']=='source_authority')==source['source_authority_sha256']
  expected=dict(path=str(cal),sha256=sha256_file(cal))
  if 'scale_prior' in record:assert record['scale_prior']==expected
  record['scale_prior']=expected
  marker.write_text(json.dumps(record,indent=2)+'\n')
  if cp['accepted_view_count']<4:raise ValueError('independent source geometry admission failed: fewer than four views')
 else:
  if not (pred/'shard_000_of_001.json').exists():raise FileNotFoundError('complete MoGe inference required before pipeline')
  manifest=root/'scene_manifest.json'
  stage('merge_geometry','radio_gs.v4.geometry.merge_moge3_shards',['--input-dir',pred,'--output',manifest],[manifest],module=True)
  cal=root/'calibration_report.json'
  stage('source_calibration','radio_gs.v4.geometry.fuse_lerf_moge3',['--scene-root',source['scene_root'],'--scene-label',a.scene,'--prediction-dir',pred,'--scene-manifest',manifest,'--output-carrier',root/'calibration_surface.pt','--output-report',cal,'--fusion-device',a.device],[cal],module=True)
 geo=root/'frozen_geometry';gp=geo/'report.json'
 stage('freeze_geometry','radio_gs.scripts.repair_moge_geometry',['--fusion-report',cal,'--expected-report-sha256',sha256_file(cal),'--output-dir',geo,'--mapping-stride','1','--skip-historical-comparator','--device',a.device]+(['--voxel-metres',a.voxel_metres] if a.voxel_metres!=.015 else [])+(['--minimum-voxel-views',a.minimum_voxel_views] if a.minimum_voxel_views!=2 else []),[gp,geo/'independent_scale.pt'],module=True)
 if a.until=='geometry':return
 gr=json.loads(gp.read_text());frames=gr['mapping_frames'];fieldroot=root/'field';sams=root/'source_sam'
 stage('source_authority','radio_gs.scripts.build_frozen_geometry_authority',['--geometry-report',gp,'--expected-report-sha256',sha256_file(gp),'--frame-ids',','.join(map(str,frames)),'--height','96','--width','128','--device',a.device,'--output-dir',fieldroot],[fieldroot/'responsibility.json',fieldroot/'source_frames.json'],module=True)
 src=json.loads((fieldroot/'source_frames.json').read_text())
 stage('source_sam','radio_gs.scripts.build_sam3_automatic_mask_cache',['--image-root',src['image_dir'],'--image-stems',','.join(Path(r['source_file']).stem for r in src['frames']),'--output-root',sams,'--device',a.device],[sams/'manifest.json'],envpy=sam,module=True)
 stage('native_teachers','radio_gs.scripts.prepare_four_teacher_retrieval',['--root',fieldroot,'--scene',a.scene,'--sam-manifest',sams/'manifest.json','--without-radio','--device',a.device],[fieldroot/'teachers.json'],module=True)
 geometry=geo/'independent_scale.pt'
 stage('query_episodes','radio_gs.scripts.compile_three_query_episodes',['--root',fieldroot,'--geometry',geometry,'--device',a.device,'--allow-missing-text-validation'],[fieldroot/'episodes.json'],module=True)
 sig=next(Path('/root/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/snapshots').iterdir())
 stage('native_query_identity','radio_gs.scripts.prepare_native_three_query_inputs',['--root',fieldroot,'--geometry',geometry,'--sam-manifest',sams/'manifest.json','--siglip-root',sig,'--dino-checkpoint','/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth','--dino-repo','/root/.cache/torch/hub/facebookresearch_dinov2_main','--allow-training-encoding-fixture'],[fieldroot/'native_query_encoder.json',fieldroot/'cold_query_fixture.pt'],module=True)
 stage('source_fusion','scripts/audits/build_source_fusion_initialization.py',['--root',fieldroot,'--output',fieldroot/'source_fusion_initialization.pt'],[fieldroot/'source_fusion_initialization.json'])
 stage('source_grounding','scripts/audits/build_source_grounded_text.py',['--root',fieldroot,'--output',fieldroot/'source_grounded_text.pt','--device',a.device],[fieldroot/'source_grounded_text.json'],envpy=sam)
 stage('grounded_supervision','scripts/audits/compile_grounded_text_supervision.py',['--root',fieldroot,'--grounding',fieldroot/'source_grounded_text.pt','--output',fieldroot/'grounded_text_supervision.pt','--device',a.device]+(['--allow-empty-text-supervision'] if a.allow_empty_text_supervision else []),[fieldroot/'grounded_text_supervision.json'])
 if a.until=='teachers':return
 from radio_gs.training.reliable_correspondence import source_region_query_tasks
 info=json.loads((fieldroot/'teachers.json').read_text());ep=torch.load(fieldroot/'episodes.pt',weights_only=False);g=torch.load(fieldroot/'grounded_text_supervision.pt',weights_only=False)
 indexed=[(i,e) for i,e in enumerate(ep['episodes']) if e['split']=='train' and e['modality']!='text']+[(len(ep['episodes'])+i,e) for i,e in enumerate(g['tasks'])]
 indexed,_=source_region_query_tasks(indexed,ep,[info['frames'].index(f) for f in info['train_frames']]);counts={m:sum(e['modality']==m for _,e in indexed) for m in ('text','image','prompt')};steps=2*sum(n>0 for n in counts.values())*max(counts.values());del ep,g,indexed
 record['training_counts']=counts;record['training_steps']=steps
 name='nvos96_protected';retained=name+'_imagecal_balanced'
 stage('field_training','radio_gs.scripts.train_three_query_field',['--root',fieldroot,'--geometry',geometry,'--name',name,'--steps',steps,'--latent-dim','96','--native-semantic-dim','64','--source-region-query-tasks','--share-siglip-adapter','--calibrated-readout','--native-decoder-readout','--source-fusion-initialization',fieldroot/'source_fusion_initialization.pt','--freeze-native-decoder','--grounded-text-supervision',fieldroot/'grounded_text_supervision.pt','--protect-native-semantic','--teachers','dino,sam,siglip','--device',a.device]+(['--allow-missing-text-tasks'] if a.allow_empty_text_supervision else []),[fieldroot/name/'report.json'],module=True)
 stage('image_calibration','scripts/audits/calibrate_frozen_source_images.py',['--root',fieldroot,'--run',name,'--device',a.device],[fieldroot/retained/'report.json'])
 if a.until=='field':return
 stage('residual_evidence','scripts/audits/build_query_free_residual_evidence.py',['--root',fieldroot],[fieldroot/'caption_region_evidence.json'])
 stage('residual_initialization','scripts/audits/fit_bounded_native_residual.py',['--root',fieldroot,'--scene',a.scene,'--parent',retained,'--device',a.device],[fieldroot/'native_residual_source_selected/report.json'])
 stage('rendered_image_residual','scripts/audits/train_rendered_native_residual.py',['--root',fieldroot,'--scene',a.scene,'--device',a.device],[fieldroot/'native_residual_rendered_selected/report.json'])
 stage('image_only_export','scripts/audits/export_image_supervised_native_residual.py',['--root',fieldroot,'--scene',a.scene,'--retained',retained],[fieldroot/'image_supervised_native_residual/report.json'])
 high=root/'authority512'
 stage('high_resolution_authority','radio_gs.scripts.build_frozen_geometry_authority',['--geometry-report',gp,'--expected-report-sha256',sha256_file(gp),'--frame-ids',','.join(map(str,frames)),'--height','384','--width','512','--device',a.device,'--output-dir',high],[high/'responsibility.json'],module=True)
 ext='image_supervised_native_residual_measured_support512';control=retained+'_measured_support512'
 stage('measured_support','scripts/audits/build_native_image_support_extension.py',['--root',fieldroot,'--authority-root',high,'--output-name',ext,'--device',a.device],[fieldroot/ext/'report.json'])
 stage('matched_support_control','scripts/audits/export_measured_support_control.py',['--root',fieldroot,'--base',retained,'--extension',ext,'--output-name',control],[fieldroot/control/'report.json'])
 # The map is query-independent; horns_center and horns_left share identical authorities.
 tasks=[a.scene] if a.scene!='horns_center' else ['horns_center','horns_left']
 for task in tasks:
  for signed in [False,True]:
   key=task+('_signed' if signed else '_image');out=root/'evaluation'/key
   stage('evaluate_'+key,'scripts/audits/evaluate_nvos_frozen_image_route.py',['--scene',task,'--root',fieldroot,'--geometry-report',gp,'--runs',retained,'image_supervised_native_residual',control,ext,'--device',a.device,'--output',out]+(['--signed-refinement'] if signed else [])+(['--signed-policy',a.signed_policy] if signed and a.signed_policy!='fixed' else [])+(['--query-root',a.query_root] if a.query_root is not None else []),[out/'report.json'],guard=False)
 record['finished']=True;marker.write_text(json.dumps(record,indent=2)+'\n');print(a.scene,'complete',flush=True)

if __name__=='__main__':main()
