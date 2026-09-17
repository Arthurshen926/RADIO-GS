"""Complete functional-teacher replacement on corrected SAM evidence."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
base=Path('output');src=base/'lerf_ropefix_multiscale_pipeline_20260917'/a.scene/'field'
# Keep resource use bounded while the structural controls are running.
while True:
 pth=src.parent/'clean_bounded_structure_validation/execution.json'
 if pth.exists():
  x=json.loads(pth.read_text())
  if any(s['status']=='failed' for s in x['stages']):raise RuntimeError('Structural control failed')
  if x.get('finished'):break
 time.sleep(10)
root=base/'clean_openclip_pipeline_20260917'/a.scene;root.mkdir(parents=True,exist_ok=a.resume);field=root/'field';gp=base/'geometry_only_repair_final_20260914'/a.scene/'report.json';geometry=Path(json.loads(gp.read_text())['results']['independent_scale']['geometry']);marker=root/'execution.json'
r=dict(scene=a.scene,source_only_training=True,independent_final_test=False,source_teacher_sha256=sha256_file(src/'teachers.json'),source_parent=str(src),stages=[],limitations=['OpenCLIP uses stored source-mask resolution for base crops; original SigLIP used native masks. Functional replacement, not a pure backbone ranking.','Teacher scale/architecture and source task counts differ.','Complete range readout and fresh online queries required before conclusions.'])
if a.resume:
 r=json.loads(marker.read_text());assert r['source_teacher_sha256']==sha256_file(src/'teachers.json')
def stage(name,entry,args,outputs,module=False,source=True):
 cmd=[sys.executable]+(['scripts/audits/run_lerf_source_stage.py',*(['--module'] if module else []),entry] if source else [*(['-m'] if module else []),entry])+list(map(str,args))
 previous=next((x for x in r['stages'] if x['name']==name),None)
 if previous is not None:
  assert previous['status']=='complete' and previous['command']==cmd,(name,'unverified resume')
  for f,h in previous['outputs'].items():assert sha256_file(f)==h
  for f,h in previous['implementations'].items():assert sha256_file(root/'sources'/name/Path(f).name)==h
  return
 impl=Path(entry.replace('.','/')+'.py') if module else Path(entry);snap=root/'sources'/name;snap.mkdir(parents=True);sources={}
 for f in [Path(__file__),impl,Path('scripts/audits/run_lerf_source_stage.py')]:shutil.copy2(f,snap/f.name);sources[str(f)]=sha256_file(f)
 cmd=[sys.executable]+(['scripts/audits/run_lerf_source_stage.py',*(['--module'] if module else []),entry] if source else [*(['-m'] if module else []),entry])+list(map(str,args));row=dict(name=name,command=cmd,implementations=sources,status='running');r['stages'].append(row);marker.write_text(json.dumps(r,indent=2)+'\n');print(a.scene,name,flush=True)
 with (root/(name+'.log')).open('x') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 row.update(status='failed' if proc.returncode else 'complete',returncode=proc.returncode)
 if not proc.returncode:row['outputs']={str(f):sha256_file(f) for f in outputs}
 marker.write_text(json.dumps(r,indent=2)+'\n')
 if proc.returncode:raise RuntimeError(name+' failed')
stage('source','scripts/audits/build_openclip_replacement_source.py',['--scene',a.scene,'--source-root',src,'--output-root',field,'--device',a.device],[field/'replacement_source.json',field/'teachers.json',field/'native_query_encoder.json'])
stage('episodes','radio_gs.scripts.compile_three_query_episodes',['--root',field,'--geometry',geometry,'--device',a.device,'--allow-missing-text-validation'],[field/'episodes.json'],module=True)
stage('initialization','scripts/audits/build_source_fusion_initialization.py',['--root',field,'--output',field/'source_fusion_initialization.pt'],[field/'source_fusion_initialization.json'])
stage('grounded_supervision','scripts/audits/compile_grounded_text_supervision.py',['--root',field,'--grounding',field/'source_grounded_text.pt','--output',field/'grounded_text_supervision.pt','--device',a.device,'--allow-empty-text-supervision'],[field/'grounded_text_supervision.json'])
from radio_gs.training.reliable_correspondence import source_region_query_tasks
info=json.loads((field/'teachers.json').read_text());ep=torch.load(field/'episodes.pt',weights_only=False);g=torch.load(field/'grounded_text_supervision.pt',weights_only=False);tasks=[(i,e) for i,e in enumerate(ep['episodes']) if e['split']=='train' and e['modality']!='text']+[(len(ep['episodes'])+i,e) for i,e in enumerate(g['tasks'])];tasks,_=source_region_query_tasks(tasks,ep,[info['frames'].index(f) for f in info['train_frames']]);counts={m:sum(e['modality']==m for _,e in tasks) for m in ('text','image','prompt')};assert all(counts.values());steps=6*max(counts.values());r.update(training_counts=counts,training_steps=steps);del tasks,ep,g
name='openclip96_protected'
stage('field','radio_gs.scripts.train_three_query_field',['--root',field,'--geometry',geometry,'--name',name,'--steps',steps,'--latent-dim',96,'--native-semantic-dim',64,'--source-region-query-tasks','--share-siglip-adapter','--calibrated-readout','--native-decoder-readout','--source-fusion-initialization',field/'source_fusion_initialization.pt','--freeze-native-decoder','--grounded-text-supervision',field/'grounded_text_supervision.pt','--protect-native-semantic','--allow-missing-text-tasks','--device',a.device],[field/name/'report.json'],module=True)
stage('image_calibration','scripts/audits/calibrate_frozen_source_images.py',['--root',field,'--run',name,'--device',a.device],[field/(name+'_imagecal_balanced')/'report.json'])
stage('caption_evidence','scripts/audits/build_functional_caption_evidence.py',['--scene',a.scene,'--source-root',src,'--root',field,'--source-groups',src/'source_phrase_groups.json','--device',a.device],[field/'caption_region_evidence.json',field/'functional_membership_groups.json'])
shutil.copy2(field/'functional_membership_groups.json',field/'source_phrase_groups.json')
stage('direct_affinity','scripts/audits/train_bounded_structure_exploration.py',['--root',field,'--geometry',geometry,'--parent',name+'_imagecal_balanced','--name','scale_affinity_v1','--variant','explicit','--passes',0,'--device',a.device],[field/'scale_affinity_v1_direct_control/report.json'])
for entry,extra,directory in [('fit_full_source_query_threshold',[],'full_source_query_threshold_control'),('fit_grounded_semantic_transfer',['--group-report','functional_membership_groups.json'],'grounded_semantic_transfer96'),('train_source_region_extents',[],'source_region_extent96'),('train_rendered_extent',['--residual-dim',128,'--steps',800],'source_extent_rendered_residual96'),('export_hierarchical_extent_candidates',[],'source_hierarchical_candidates96'),('export_native_support_overlap',['--fallback-only'],'native_overlap_fallback96'),('fit_source_scope_gate',['--hard-pairs'],'hard_source_scope_gate96'),('export_rank_preserving_completion',['--within-extent'],'native_within_extent96'),('fit_exact_render_query_threshold',[],'exact_render_query_threshold96')]:
 stage(entry,'scripts/audits/'+entry+'.py',['--root',field,'--scene',a.scene,*extra,*(['--device',a.device] if not entry.startswith('export_') else [])],[field/directory/'report.json'])
stage('precision_gate','scripts/audits/calibrate_native_gate_precision.py',['--root',field,'--device',a.device],[field/'source_precision_gate96/report.json'])
stage('source_gate_audit','scripts/audits/audit_source_gate_stability.py',['--scene',a.scene,'--root',field,'--device',a.device,'--output',root/'source_gate_audit.json'],[root/'source_gate_audit.json'])
for task in ('text','reference'):
 stage('evaluate_'+task,'scripts/audits/evaluate_functional_queries.py',['--task',task,'--root',field,'--scene',a.scene,'--geometry-report',gp,'--runs','source_precision_gate96',name+'_imagecal_balanced','--device',a.device,'--output',root/(task+'_evaluation.json')],[root/(task+'_evaluation.json')],source=False)
r['finished']=True;marker.write_text(json.dumps(r,indent=2)+'\n')
