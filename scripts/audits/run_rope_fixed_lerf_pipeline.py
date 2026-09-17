"""Fresh single-grid SAM source inventory and complete native-preserving field.

Same source RGB/cameras/geometry as the retained development control. Unlike
the old mixed multiscale/grounded inventory, automatic seeds use the same clean
grid12 recipe as NVOS; this is a complete recipe comparison, not a pure cast
ablation. No old masks, embeddings, correspondence graph or field is reused.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=base/'lerf_ropefix_pipeline_20260917'/a.scene
    root.mkdir(parents=True,exist_ok=a.resume);fieldroot=root/'field'
    old=base/'preserved_source_inventory_20260916'/a.scene;oldinfo=json.loads((old/'teachers.json').read_text())
    gp=base/'geometry_only_repair_final_20260914'/a.scene/'report.json';gr=json.loads(gp.read_text());geometry=Path(gr['results']['independent_scale']['geometry'])
    py=sys.executable;sam='/root/miniconda3/envs/sam3/bin/python';record=dict(scene=a.scene,source_only_reconstruction=True,old_geometry_report_sha256=sha256_file(gp),source_frames=oldinfo['frames'],validation_frames=oldinfo['validation_frames'],source_inventory_policy='Fresh corrected SAM3 single grid12; replaces old mixed multiscale/grounded inventory; not a pure precision ablation.',stages=[])
    marker=root/'execution.json'
    if marker.exists():
        previous=json.loads(marker.read_text())
        for key in ['scene','old_geometry_report_sha256','source_frames','validation_frames']:assert previous[key]==record[key]
        record=previous
    def stage(name,entry,args,outputs,*,module=False,envpy=py,source=True):
        implementation=Path(entry.replace('.','/')+'.py') if module else Path(entry)
        command=[envpy]
        if source:command+=['scripts/audits/run_lerf_source_stage.py']+(['--module'] if module else [])+[entry]
        else:command+=([ '-m',entry] if module else [entry])
        command+=list(map(str,args));sources=[Path(__file__),implementation,Path('scripts/audits/run_lerf_source_stage.py')]
        if name in ('source_sam','source_grounding','caption_grounding'):sources += [Path('radio_gs/scripts/build_sam3_foundation_cache.py'),Path('/root/external/sam3/sam3/model_builder.py'),Path('/root/external/sam3/sam3/model/vitdet.py')]
        old=next((s for s in record['stages'] if s['name']==name),None)
        if old is not None:
            assert old['command']==command,'stage command changed: '+name
            if old['status']=='complete':
                for f,h in old['outputs'].items():assert sha256_file(f)==h
                for f,h in old['implementations'].items():assert sha256_file(root/'sources'/name/Path(f).name)==h
                return
            raise RuntimeError('Incomplete stage requires explicit inspection: '+name)
        snap=root/'sources'/name;snap.mkdir(parents=True)
        identities={}
        for f in sources:shutil.copy2(f,snap/f.name);identities[str(f)]=sha256_file(f)
        row=dict(name=name,command=command,implementations=identities,status='running');record['stages'].append(row);marker.write_text(json.dumps(record,indent=2)+'\n');print(a.scene,name,flush=True)
        with (root/(name+'.log')).open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH='.'))
        row.update(returncode=r.returncode,status='failed' if r.returncode else 'complete')
        if not r.returncode:row['outputs']={str(f):sha256_file(f) for f in outputs}
        marker.write_text(json.dumps(record,indent=2)+'\n')
        if r.returncode:raise RuntimeError(name+' failed')
    stage('authority','radio_gs.scripts.build_frozen_geometry_authority',['--geometry-report',gp,'--expected-report-sha256',sha256_file(gp),'--frame-ids',','.join(map(str,oldinfo['frames'])),'--height','96','--width','128','--device',a.device,'--output-dir',fieldroot],[fieldroot/'responsibility.json'],module=True)
    src=json.loads((fieldroot/'source_frames.json').read_text());sams=root/'source_sam'
    stage('source_sam','radio_gs.scripts.build_sam3_automatic_mask_cache',['--image-root',src['image_dir'],'--image-stems',','.join(Path(r['source_file']).stem for r in src['frames']),'--output-root',sams,'--device',a.device],[sams/'manifest.json'],module=True,envpy=sam)
    stage('teachers','radio_gs.scripts.prepare_four_teacher_retrieval',['--root',fieldroot,'--scene',a.scene,'--sam-manifest',sams/'manifest.json',*(['--radio-root',base/'canonical_teacher_features/ramen_fix4c_exact120_v1'] if a.scene=='ramen' else []),'--validation-frame-ids',*oldinfo['validation_frames'],'--device',a.device],[fieldroot/'teachers.json'],module=True)
    stage('episodes','radio_gs.scripts.compile_three_query_episodes',['--root',fieldroot,'--geometry',geometry,'--device',a.device],[fieldroot/'episodes.json'],module=True)
    sig=next(Path('/root/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/snapshots').iterdir())
    stage('native_queries','radio_gs.scripts.prepare_native_three_query_inputs',['--root',fieldroot,'--geometry',geometry,'--sam-manifest',sams/'manifest.json','--siglip-root',sig,'--dino-checkpoint','/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth','--dino-repo','/root/.cache/torch/hub/facebookresearch_dinov2_main','--allow-training-encoding-fixture'],[fieldroot/'native_query_encoder.json',fieldroot/'cold_query_fixture.pt'],module=True)
    stage('source_fusion','scripts/audits/build_source_fusion_initialization.py',['--root',fieldroot,'--output',fieldroot/'source_fusion_initialization.pt'],[fieldroot/'source_fusion_initialization.json'])
    stage('source_grounding','scripts/audits/build_source_grounded_text.py',['--root',fieldroot,'--output',fieldroot/'source_grounded_text.pt','--device',a.device],[fieldroot/'source_grounded_text.json'],envpy=sam)
    stage('grounded_supervision','scripts/audits/compile_grounded_text_supervision.py',['--root',fieldroot,'--grounding',fieldroot/'source_grounded_text.pt','--output',fieldroot/'grounded_text_supervision.pt','--device',a.device],[fieldroot/'grounded_text_supervision.json'])
    from radio_gs.training.reliable_correspondence import source_region_query_tasks
    info=json.loads((fieldroot/'teachers.json').read_text());ep=torch.load(fieldroot/'episodes.pt',weights_only=False);g=torch.load(fieldroot/'grounded_text_supervision.pt',weights_only=False)
    tasks=[(i,e) for i,e in enumerate(ep['episodes']) if e['split']=='train' and e['modality']!='text']+[(len(ep['episodes'])+i,e) for i,e in enumerate(g['tasks'])]
    tasks,_=source_region_query_tasks(tasks,ep,[info['frames'].index(f) for f in info['train_frames']]);counts={m:sum(e['modality']==m for _,e in tasks) for m in ('text','image','prompt')};assert all(counts.values());steps=6*max(counts.values());record.update(training_counts=counts,training_steps=steps);del ep,g,tasks
    name='ropefix96_protected';parent=name+'_imagecal_balanced'
    stage('field','radio_gs.scripts.train_three_query_field',['--root',fieldroot,'--geometry',geometry,'--name',name,'--steps',steps,'--latent-dim','96','--native-semantic-dim','64','--source-region-query-tasks','--share-siglip-adapter','--calibrated-readout','--native-decoder-readout','--source-fusion-initialization',fieldroot/'source_fusion_initialization.pt','--freeze-native-decoder','--grounded-text-supervision',fieldroot/'grounded_text_supervision.pt','--protect-native-semantic','--device',a.device],[fieldroot/name/'report.json'],module=True)
    stage('image_calibration','scripts/audits/calibrate_frozen_source_images.py',['--root',fieldroot,'--run',name,'--device',a.device],[fieldroot/parent/'report.json'])
    for name,entry,out,env in [('caption_discovery','discover_source_caption_phrases','caption_source_phrases.json',py),('caption_grounding','ground_source_caption_phrases','caption_grounded_regions.json',sam),('caption_encoding','encode_caption_region_evidence','caption_region_evidence.json',py)]:
        stage(name,'scripts/audits/'+entry+'.py',['--root',fieldroot,'--scene',a.scene,'--device',a.device],[fieldroot/out],envpy=env)
    stage('direct_affinity','scripts/audits/train_bounded_structure_exploration.py',['--root',fieldroot,'--geometry',geometry,'--parent',parent,'--variant','explicit','--passes','0','--name','scale_affinity_v1','--device',a.device],[fieldroot/'scale_affinity_v1_direct_control/report.json'])
    stage('phrase_groups','scripts/audits/build_source_phrase_groups.py',['--root',fieldroot],[fieldroot/'source_phrase_groups.json'])
    stages=[('fit_full_source_query_threshold',[],'full_source_query_threshold_control'),('fit_grounded_semantic_transfer',['--group-report','source_phrase_groups.json'],'grounded_semantic_transfer96'),('train_source_region_extents',[],'source_region_extent96'),('train_rendered_extent',['--residual-dim','128','--steps','800'],'source_extent_rendered_residual96'),('export_hierarchical_extent_candidates',[],'source_hierarchical_candidates96'),('export_native_support_overlap',['--fallback-only'],'native_overlap_fallback96'),('fit_source_scope_gate',['--hard-pairs'],'hard_source_scope_gate96'),('export_rank_preserving_completion',['--within-extent'],'native_within_extent96'),('fit_exact_render_query_threshold',[],'exact_render_query_threshold96')]
    for entry,extra,directory in stages:
        command=['--root',fieldroot,'--scene',a.scene,*extra]
        if not entry.startswith('export_'):command+=['--device',a.device]
        stage(entry,'scripts/audits/'+entry+'.py',command,[fieldroot/directory/'report.json'])
    final_field=fieldroot/'exact_render_query_threshold96/field.pt'
    stage('cold_native_queries','scripts/audits/cold_three_query_native.py',['--fixture',fieldroot/'cold_query_fixture.pt','--field',final_field,'--field-sha256',sha256_file(final_field),'--device',a.device,'--output',root/'cold_native_queries.json'],[root/'cold_native_queries.json'])
    for task in ['text','reference']:
        out=root/(task+'_evaluation.json')
        stage('evaluate_'+task,'scripts/audits/evaluate_functional_queries.py',['--task',task,'--root',fieldroot,'--scene',a.scene,'--geometry-report',gp,'--runs','exact_render_query_threshold96',parent,'--device',a.device,'--output',out],[out],source=False)
    record['finished']=True;marker.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
