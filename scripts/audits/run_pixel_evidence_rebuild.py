"""Fixed source-evidence fusion control with complete dependent readout refitting."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args()
    root=Path('output/pixel_evidence_rebuild_20260917')/a.scene;root.mkdir(parents=True,exist_ok=False);fieldroot=root/'field';evidence=Path('output/lerf_pose_covered_pipeline_20260917')/a.scene/'field';gp=Path('output/geometry_only_repair_final_20260914')/a.scene/'report.json'
    record=dict(scene=a.scene,source_only_training=True,benchmark_selected_fusion=False,selection='Pixel partition improves both all-view and leave-view-out source pseudo-label AP in both scenes; no benchmark query used to select weighting.',parent=str(evidence),stages=[],field_structural_training_repeated=False,all_field_dependent_range_heads_refitted=True)
    marker=root/'execution.json'
    def stage(name,entry,args,outputs,source=True):
        cmd=[sys.executable]+(['scripts/audits/run_lerf_source_stage.py',entry] if source else [entry])+list(map(str,args));snap=root/'sources'/name;snap.mkdir(parents=True);identities={}
        for f in [Path(__file__),Path(entry),Path('radio_gs/training/source_pixel_evidence.py'),Path('scripts/audits/run_lerf_source_stage.py')]:shutil.copy2(f,snap/f.name);identities[str(f)]=sha256_file(f)
        row=dict(name=name,command=cmd,implementations=identities,status='running');record['stages'].append(row);marker.write_text(json.dumps(record,indent=2)+'\n');print(a.scene,name,flush=True)
        with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        row.update(status='failed' if r.returncode else 'complete',returncode=r.returncode)
        if not r.returncode:row['outputs']={str(f):sha256_file(f) for f in outputs}
        marker.write_text(json.dumps(record,indent=2)+'\n')
        if r.returncode:raise RuntimeError(name+' failed')
    stage('prepare','scripts/audits/prepare_pixel_evidence_reconstruction.py',['--parent',evidence,'--output',fieldroot,'--device',a.device],[fieldroot/'prepared_inputs.json',fieldroot/'pixel_evidence96/report.json'])
    parent='pixel_evidence96_imagecal_balanced'
    stage('image_calibration','scripts/audits/calibrate_frozen_source_images.py',['--root',fieldroot,'--run','pixel_evidence96','--device',a.device],[fieldroot/parent/'report.json'])
    shutil.copytree(fieldroot/parent,fieldroot/'scale_affinity_v1_direct_control')
    record['affinity_copy_sha256']=sha256_file(fieldroot/'scale_affinity_v1_direct_control/field.pt')
    stages=[('fit_full_source_query_threshold',[],'full_source_query_threshold_control'),('fit_grounded_semantic_transfer',['--group-report','source_phrase_groups.json'],'grounded_semantic_transfer96'),('train_source_region_extents',[],'source_region_extent96'),('train_rendered_extent',['--residual-dim','128','--steps','800'],'source_extent_rendered_residual96'),('export_hierarchical_extent_candidates',[],'source_hierarchical_candidates96'),('export_native_support_overlap',['--fallback-only'],'native_overlap_fallback96'),('fit_source_scope_gate',['--hard-pairs'],'hard_source_scope_gate96'),('export_rank_preserving_completion',['--within-extent'],'native_within_extent96'),('fit_exact_render_query_threshold',[],'exact_render_query_threshold96')]
    for entry,extra,directory in stages:
        command=['--root',fieldroot,'--scene',a.scene,*extra]
        if not entry.startswith('export_'):command+=['--device',a.device]
        stage(entry,'scripts/audits/'+entry+'.py',command,[fieldroot/directory/'report.json'])
    stage('precision_gate','scripts/audits/calibrate_native_gate_precision.py',['--root',fieldroot,'--device',a.device],[fieldroot/'source_precision_gate96/report.json'])
    stage('joint_native_residual','scripts/audits/train_joint_native_retrieval_residual.py',['--root',fieldroot,'--device',a.device,'--parent','source_precision_gate96'],[fieldroot/'joint_native_retrieval_residual/report.json'])
    stage('scoped_image_residual','scripts/audits/export_image_supervised_native_residual.py',['--root',fieldroot,'--scene',a.scene,'--retained','source_precision_gate96','--learned-parent','joint_native_retrieval_residual','--name','joint_supervised_image_residual'],[fieldroot/'joint_supervised_image_residual/report.json'])
    final_field=fieldroot/'joint_supervised_image_residual/field.pt'
    stage('cold_native_queries','scripts/audits/cold_three_query_native.py',['--fixture',fieldroot/'cold_query_fixture.pt','--field',final_field,'--field-sha256',sha256_file(final_field),'--device',a.device,'--require-concept-route-consistency','--output',root/'cold_native_queries.json'],[root/'cold_native_queries.json'])
    for task in ['text','reference']:
        out=root/(task+'_evaluation.json')
        stage('evaluate_'+task,'scripts/audits/evaluate_functional_queries.py',['--task',task,'--root',fieldroot,'--scene',a.scene,'--geometry-report',gp,'--runs','joint_supervised_image_residual','source_precision_gate96',parent,'--device',a.device,'--output',out],[out],source=False)
    record['finished']=True;marker.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
