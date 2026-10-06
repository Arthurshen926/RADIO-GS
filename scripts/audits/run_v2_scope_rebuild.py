"""Rebuild every caption-dependent asset, reusing verified independent base state."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 old=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';root=Path('output/v2_scope_rebuild_20260918')/a.scene;field=root/'field';marker=root/'execution.json';sam='/root/miniconda3/envs/sam3/bin/python';gp=Path('output/geometry_only_repair_final_20260914')/a.scene/'report.json'
 if marker.exists():
  if not a.resume:raise FileExistsError(marker)
  record=json.loads(marker.read_text())
  for f,h in record['reused_input_hashes'].items():assert sha256_file(f)==h
 else:
  field.mkdir(parents=True,exist_ok=False);inputs={}
  # Caption generation follows base structural training. It is not an input to
  # native pixel fusion, base episodes or image calibration. Only these proven
  # independent assets are reused; no caption/extent outputs are copied.
  names=['teachers.json','episodes.json','episodes.pt','responsibility.json','source_frames.json','native_query_encoder.json','queries.pt','cold_query_fixture.pt','cold_query_crop.png','grounded_text_supervision.pt','grounded_text_supervision.json','source_grounded_text.pt','source_grounded_text.json','caption_source_phrases.json']
  info=json.loads((old/'teachers.json').read_text());names += [x['path'] for x in info['teacher_files'].values()]
  for name in sorted(set(names)):
   src=old/name;dst=field/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src.resolve(),dst);inputs[str(src)]=sha256_file(src);assert sha256_file(dst)==inputs[str(src)]
  for name in ['responsibility.json.views','scale_affinity_v1_direct_control']:
   shutil.copytree(old/name,field/name,copy_function=shutil.copy2)
   for f in (old/name).rglob('*'):
    if f.is_file():inputs[str(f)]=sha256_file(f)
  record=dict(schema='v2_scope_rebuild.v1',scene=a.scene,source_only_training=True,reused_input_hashes=inputs,reused_base_reason='Base geometry, SAM episode inventory, protected structure, pixel-native fusion and initial image calibration do not consume caption masks. All caption-dependent assets regenerated.',v2_native_scope=True,steps=[]);marker.write_text(json.dumps(record,indent=2)+'\n')
 def stage(name,entry,args,outputs,envpy=sys.executable,source=True):
  command=[envpy]+(['scripts/audits/run_lerf_source_stage.py',entry] if source else [entry])+list(map(str,args));previous=next((r for r in record['steps'] if r['name']==name),None)
  if previous:
   assert previous['command']==command
   if previous['status']!='complete':raise RuntimeError('Inspect incomplete stage before resuming: '+name)
   for f,h in previous['outputs'].items():assert sha256_file(f)==h
   return
  snap=root/'sources'/name;snap.mkdir(parents=True);impl={}
  paths=[Path(__file__),Path(entry),Path('radio_gs/utils/native_scope_identity.py'),Path('radio_gs/training/scope_union_lifting.py')]
  for f in paths:shutil.copy2(f,snap/f.name);impl[str(f)]=sha256_file(f)
  row=dict(name=name,command=command,implementations=impl,status='running');record['steps'].append(row);marker.write_text(json.dumps(record,indent=2)+'\n');print(a.scene,name,flush=True)
  with (root/(name+'.log')).open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH='.'))
  row.update(status='complete' if r.returncode==0 else 'failed',returncode=r.returncode)
  if r.returncode==0:row['outputs']={str(f):sha256_file(f) for f in outputs}
  marker.write_text(json.dumps(record,indent=2)+'\n')
  if r.returncode:raise RuntimeError(name+' failed')
 stage('grounding','scripts/audits/ground_source_caption_phrases.py',['--scene',a.scene,'--root',field,'--device',a.device],[field/'caption_grounded_regions.pt',field/'caption_grounded_regions.json'],envpy=sam)
 stage('encoding','scripts/audits/encode_caption_region_evidence.py',['--scene',a.scene,'--root',field,'--device',a.device],[field/'caption_region_evidence.pt',field/'caption_region_evidence.json'])
 stage('groups','scripts/audits/build_source_phrase_groups.py',['--root',field],[field/'source_phrase_groups.json'])
 stages=[('fit_full_source_query_threshold',[],'full_source_query_threshold_control'),('fit_grounded_semantic_transfer',['--group-report','source_phrase_groups.json'],'grounded_semantic_transfer96'),('train_source_region_extents',[],'source_region_extent96'),('train_rendered_extent',['--residual-dim','128','--steps','800'],'source_extent_rendered_residual96'),('export_hierarchical_extent_candidates',[],'source_hierarchical_candidates96'),('export_native_support_overlap',['--fallback-only'],'native_overlap_fallback96'),('fit_source_scope_gate',['--hard-pairs'],'hard_source_scope_gate96'),('export_rank_preserving_completion',['--within-extent'],'native_within_extent96'),('fit_exact_render_query_threshold',[],'exact_render_query_threshold96')]
 for entry,extra,directory in stages:
  args=['--root',field,'--scene',a.scene,*extra]
  if not entry.startswith('export_'):args+=['--device',a.device]
  stage(entry,'scripts/audits/'+entry+'.py',args,[field/directory/'report.json'])
 stage('precision','scripts/audits/calibrate_native_gate_precision.py',['--root',field,'--device',a.device],[field/'source_precision_gate96/report.json'])
 stage('joint_residual','scripts/audits/train_joint_native_retrieval_residual.py',['--root',field,'--device',a.device,'--parent','source_precision_gate96'],[field/'joint_native_retrieval_residual/report.json'])
 stage('image_residual','scripts/audits/export_image_supervised_native_residual.py',['--root',field,'--scene',a.scene,'--retained','source_precision_gate96','--learned-parent','joint_native_retrieval_residual','--name','joint_supervised_image_residual'],[field/'joint_supervised_image_residual/report.json'])
 final='counterfactual_extent_benefit_admission'
 stage('admission','scripts/audits/fit_counterfactual_extent_admission.py',['--root',field,'--scene',a.scene,'--device',a.device],[field/final/'report.json'])
 for task in ['text','reference']:
  stage('evaluate_'+task,'scripts/audits/audit_current_query_routes.py',['--task',task,'--root',field,'--scene',a.scene,'--geometry-report',gp,'--runs',final,'--device',a.device,'--output',root/(task+'_online.json')],[root/(task+'_online.json'),root/(task+'_online.routes.json')],source=False)
  stage('replay_'+task,'scripts/audits/audit_current_query_routes.py',['--task',task,'--encoding-replay',root/(task+'_online.encodings.pt'),'--root',field,'--scene',a.scene,'--geometry-report',gp,'--runs',final,'--device',a.device,'--output',root/(task+'_replay.json')],[root/(task+'_replay.json')],source=False)
 record['finished']=True;marker.write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
