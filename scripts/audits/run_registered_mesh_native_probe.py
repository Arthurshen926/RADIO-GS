"""Registered ScanNet dataset adapter for the frozen source-field recipe."""
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
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--output-root',type=Path);p.add_argument('--geometry-report',type=Path);p.add_argument('--until-base',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    if not a.until_base:raise ValueError('Registered adapter exposes the source base only; use registered pixel runner next')
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=a.output_root or base/'scannet_frozen_field_20260918'/a.scene/'base'
    root.mkdir(parents=True,exist_ok=True);fieldroot=root/'raw_field'
    sam_execution=json.loads((root/'source_sam_execution.json').read_text());assert sam_execution['status']=='complete'
    for f,h in sam_execution['implementations'].items():assert sha256_file(root/'sam_sources'/Path(f).name)==h
    old=base/'preserved_source_inventory_20260916'/a.scene;oldinfo=json.loads((root/'source_plan.json').read_text())
    gp=a.geometry_report or base/'scannet_frozen_field_20260918'/a.scene/'geometry_report.json';gr=json.loads(gp.read_text());geometry=Path(gr['results']['independent_scale']['geometry'])
    py=sys.executable;sam='/root/miniconda3/envs/sam3/bin/python';record=dict(scene=a.scene,source_only_reconstruction=True,source_plan_sha256=sha256_file(root/'source_plan.json'),old_geometry_report_sha256=sha256_file(gp),source_frames=oldinfo['frames'],validation_frames=oldinfo['validation_frames'],source_inventory_policy='Same corrected multiscale and grounded inventory recipe, 16 training views selected by predeclared geometry coverage. Stop after native source fusion; no full field or readout training.',source_sam_execution_sha256=sha256_file(root/'source_sam_execution.json'),source_sam_manifest_sha256=sha256_file(root/'source_sam/manifest.json'),stages=[])
    marker=root/'execution.json'
    if marker.exists():
        previous=json.loads(marker.read_text())
        for key in ['scene','old_geometry_report_sha256','source_frames','validation_frames']:assert previous[key]==record[key]
        record=previous
    def stage(name,entry,args,outputs,*,module=False,envpy=py,source=True):
        implementation=Path(entry.replace('.','/')+'.py') if module else Path(entry)
        command=[envpy]
        if source:command+=['scripts/audits/run_scannet_source_stage.py']+(['--module'] if module else [])+[entry]
        else:command+=([ '-m',entry] if module else [entry])
        command+=list(map(str,args));sources=[Path(__file__),implementation,Path('scripts/audits/run_scannet_source_stage.py')]
        if name in ('source_sam','raw_source_grounding','caption_grounding'):sources += [Path('radio_gs/scripts/build_sam3_foundation_cache.py'),Path('/root/external/sam3/sam3/model_builder.py'),Path('/root/external/sam3/sam3/model/vitdet.py')]
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
    stage('authority','scripts/audits/build_registered_mesh_authority.py',['--geometry-report',gp,'--expected-report-sha256',sha256_file(gp),'--frame-ids',','.join(map(str,oldinfo['frames'])),'--height','96','--width','128','--device',a.device,'--output-dir',fieldroot],[fieldroot/'responsibility.json'],module=False)
    src=json.loads((fieldroot/'source_frames.json').read_text());sams=root/'source_sam'
    stage('teachers','scripts/audits/prepare_registered_teacher_evidence.py',['--geometry',geometry,'--root',fieldroot,'--scene',a.scene,'--sam-manifest',sams/'manifest.json','--radio-root',oldinfo['radio_root'],'--validation-frame-ids',*oldinfo['validation_frames'],'--device',a.device],[fieldroot/'teachers.json'],module=False)
    effective=json.loads((fieldroot/'teachers.json').read_text());record['effective_train_frames']=effective['train_frames'];record['effective_validation_frames']=effective['validation_frames']
    stage('episodes','radio_gs.scripts.compile_three_query_episodes',['--root',fieldroot,'--geometry',geometry,'--device',a.device],[fieldroot/'episodes.json'],module=True)
    sig=next(Path('/root/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/snapshots').iterdir())
    stage('native_queries','radio_gs.scripts.prepare_native_three_query_inputs',['--root',fieldroot,'--geometry',geometry,'--sam-manifest',sams/'manifest.json','--siglip-root',sig,'--dino-checkpoint','/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth','--dino-repo','/root/.cache/torch/hub/facebookresearch_dinov2_main','--allow-training-encoding-fixture'],[fieldroot/'native_query_encoder.json',fieldroot/'cold_query_fixture.pt'],module=True)
    stage('raw_source_grounding','scripts/audits/build_source_grounded_text.py',['--root',fieldroot,'--output',fieldroot/'source_grounded_text.pt','--device',a.device],[fieldroot/'source_grounded_text.json'],envpy=sam)
    stage('grounded_inventory','scripts/audits/build_grounded_native_region_inventory.py',['--root',fieldroot,'--grounding',fieldroot/'source_grounded_text.pt','--output',fieldroot/'grounded_region_inventory.pt','--split-components','--device',a.device],[fieldroot/'grounded_region_inventory.json'])
    stage('augment_inventory','scripts/audits/prepare_component_source_root.py',['--parent',fieldroot,'--inventory',fieldroot/'grounded_region_inventory.pt','--output',root/'field'],[root/'field/teachers.json'])
    rawroot=fieldroot;fieldroot=root/'field'
    stage('augmented_episodes','radio_gs.scripts.compile_three_query_episodes',['--root',fieldroot,'--geometry',geometry,'--device',a.device],[fieldroot/'episodes.json'],module=True)
    stage('rebind_grounding','scripts/audits/rebind_source_grounding_inventory.py',['--parent',rawroot,'--root',fieldroot],[fieldroot/'source_grounded_text.json'])
    stage('source_fusion','scripts/audits/build_source_fusion_initialization.py',['--root',fieldroot,'--output',fieldroot/'source_fusion_initialization.pt'],[fieldroot/'source_fusion_initialization.json'])
    record.update(native_inputs_ready=True,full_field_training_performed=False,native_classification_only=True);marker.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
