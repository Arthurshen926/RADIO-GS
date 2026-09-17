"""Verify completed clean-source receipts and expose missing work explicitly."""
import argparse
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


def summarize(base):
    result={'schema':'ropefixed_rebuild_summary.v1','independent_final_test':False,
            'complete_joint_benchmark':False,'nvos':{},'lerf':{},'lerf_multiscale':{},'lerf_view_balanced':{}}
    for benchmark,dirname,scenes in [('nvos','nvos_ropefix_pipeline_20260917',['fern','flower','fortress','trex','orchids','horns_center','leaves']),('lerf','lerf_ropefix_pipeline_20260917',['figurines','ramen']),('lerf_multiscale','lerf_ropefix_multiscale_pipeline_20260917',['figurines','ramen']),('lerf_view_balanced','lerf_view_balanced_rebuild_20260917',['figurines','ramen'])]:
        for scene in scenes:
            root=base/dirname/scene
            marker=root/'execution.json'
            if not marker.exists():result[benchmark][scene]={'status':'not_started'};continue
            execution=json.loads(marker.read_text())
            if benchmark=='lerf_multiscale':
                sp=root/'source_sam_execution.json'
                assert sha256_file(sp)==execution['source_sam_execution_sha256']
                assert sha256_file(root/'source_sam/manifest.json')==execution['source_sam_manifest_sha256']
                for f,digest in json.loads(sp.read_text())['implementations'].items():assert sha256_file(root/'sam_sources'/Path(f).name)==digest
            stages=execution['stages'];stages=list(stages.items()) if isinstance(stages,dict) else [(s['name'],s) for s in stages]
            row={'status':'complete' if execution.get('finished') else 'incomplete','execution_sha256':sha256_file(marker),'stages':{k:s['status'] for k,s in stages}}
            if benchmark=='lerf_view_balanced' and (root/'field/prepared_inputs.json').exists():
                prepared=json.loads((root/'field/prepared_inputs.json').read_text())
                for f,h in prepared['input_hashes'].items():assert sha256_file(root/'field'/f)==h,(scene,f)
                assert sha256_file(root/'field/teachers.json')==execution['source_teacher_sha256']
                init=root/'field/source_fusion_initialization.pt'
                init_receipt=json.loads(init.with_suffix('.json').read_text())
                assert sha256_file(init)==init_receipt['sha256']
                assert init_receipt['policy']=='equal_available_source_views'
                row['source_fusion_policy']=init_receipt['policy']
            for name,s in stages:
                if s['status']!='complete':continue
                for f,h in s['outputs'].items():assert sha256_file(f)==h,(scene,name,f)
                for f,h in s['implementations'].items():assert sha256_file(root/'sources'/name/Path(f).name)==h,(scene,name,f)
            manifest=root/'source_sam/manifest.json'
            if manifest.exists():
                c=json.loads(manifest.read_text())['generation_contract']
                assert c['complex_rope_buffers_preserved']
                if benchmark!='lerf_multiscale':assert c['schema_version']>=3
                row['source_sam_contract']=c
            row['evaluations']={}
            if benchmark=='nvos':
                for p in sorted((root/'evaluation').glob('*/report.json')):
                    e=json.loads(p.read_text())
                    assert e['sam_reference_complex_rope_preserved'] and not e['target_rgb_opened'] and e['target_gt_used_after_prediction_only']
                    for name,digest in e['fields'].items():assert sha256_file(root/'field'/name/'field.pt')==digest
                    for name,metrics in e['results'].items():assert sha256_file(p.parent/(name+'.pt'))==metrics['prediction_sha256']
                    row['evaluations'][p.parent.name]={'report_sha256':sha256_file(p),'metrics':e['results'],'signed_constraints':e.get('refinement_records',{})}
            else:
                for task in ['text','reference']:
                    p=root/(task+'_evaluation.json')
                    if p.exists():
                        e=json.loads(p.read_text())
                        for name,v in e['runs'].items():assert sha256_file(root/'field'/name/'field.pt')==v['field_sha256']
                        dispatch=json.loads(p.with_suffix('.dispatch.json').read_text())
                        assert sha256_file(p)==dispatch['result_sha256']
                        assert sha256_file(p.with_suffix('.protocol.json'))==dispatch['protocol_sha256']
                        row['evaluations'][task]={'report_sha256':sha256_file(p),'results':e}
                if benchmark in ('lerf_multiscale','lerf_view_balanced'):
                    validation=root/'precision_gate_validation'
                    row['precision_validation_status']='not_started'
                    if (validation/'execution.json').exists():
                        vr=json.loads((validation/'execution.json').read_text())
                        row['precision_validation_status']='complete' if vr.get('finished') else 'incomplete'
                        for f,h in vr['implementations'].items():assert sha256_file(validation/'sources'/Path(f).name)==h
                        for stage in vr['stages']:
                            if stage['status']=='complete':
                                for f,h in stage['outputs'].items():assert sha256_file(f)==h
                        if vr.get('finished'):
                            repair=root/'field/source_precision_gate96/report.json'
                            assert sha256_file(repair)==vr['source_repair_report_sha256']
                            pr=json.loads(repair.read_text())
                            assert sha256_file(repair.parent/'field.pt')==pr['field_sha256']
                            assert sha256_file(repair.parent/'source.py')==pr['implementation_sha256']
                            cold=json.loads((validation/'cold.json').read_text())
                            assert cold['modalities']['text']['concept_route_consistent']
                            row['precision_repair']={'report_sha256':sha256_file(repair),'cold':cold,'evaluations':{}}
                            for task in ('text','reference'):
                                v=json.loads((validation/(task+'.json')).read_text())
                                assert v['runs']['source_precision_gate96']['field_sha256']==pr['field_sha256']
                                row['precision_repair']['evaluations'][task]=v
                if benchmark=='lerf_multiscale':
                    row['joint_controls']={}
                    for folder in ('clean_joint_validation','scoped_joint_image_validation'):
                        directory=root/folder;receipt=directory/'execution.json'
                        if not receipt.exists():row['joint_controls'][folder]={'status':'not_started'};continue
                        vr=json.loads(receipt.read_text());item={'status':'complete' if vr.get('finished') else 'incomplete','execution_sha256':sha256_file(receipt)}
                        sources=vr.get('training_sources',vr.get('implementations',{}))
                        for f,h in sources.items():assert sha256_file(directory/'sources'/Path(f).name)==h
                        for stage in vr['stages']:
                            for f,h in stage.get('implementations',{}).items():assert sha256_file(directory/'sources'/stage['name']/Path(f).name)==h
                            if stage['status']=='complete':
                                for f,h in stage['outputs'].items():assert sha256_file(f)==h
                        if vr.get('finished'):
                            item['evaluations']={}
                            for task in ('text','reference'):
                                v=json.loads((directory/(task+'.json')).read_text())
                                for name,metrics in v['runs'].items():assert sha256_file(root/'field'/name/'field.pt')==metrics['field_sha256']
                                item['evaluations'][task]=v
                        row['joint_controls'][folder]=item
            result[benchmark][scene]=row
    result['all_scheduled_rebuilds_finished']=all(r['status']=='complete' for b in ['nvos','lerf','lerf_multiscale','lerf_view_balanced'] for r in result[b].values())
    result['all_precision_validations_finished']=all(r.get('precision_validation_status')=='complete' for b in ['lerf_multiscale','lerf_view_balanced'] for r in result[b].values())
    result['all_clean_joint_controls_finished']=all(len(r.get('joint_controls',{}))==2 and all(v['status']=='complete' for v in r['joint_controls'].values()) for r in result['lerf_multiscale'].values())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path('/mnt/pool/sqy/results/RADIO-GS/output'));p.add_argument('--output',type=Path,required=True);p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=summarize(a.base)
    if a.require_complete and not (result['all_scheduled_rebuilds_finished'] and result['all_precision_validations_finished'] and result['all_clean_joint_controls_finished']):raise RuntimeError('Scheduled rebuilds or bounded validations remain incomplete')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print({b:{s:r['status'] for s,r in result[b].items()} for b in ['nvos','lerf','lerf_multiscale','lerf_view_balanced']})

if __name__=='__main__':main()
