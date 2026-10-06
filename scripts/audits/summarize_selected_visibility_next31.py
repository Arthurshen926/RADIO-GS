"""Bound fixed-map paired outcomes; correct inherited reconstruction metadata."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

torch.set_num_threads(4)
base=Path('output/selected_visibility_next31_v2')
registration=json.loads((base/'registration.json').read_text())
evaluation=json.loads((base/'evaluation_registration.json').read_text())
assert registration['implementation_sha256']==sha256_file('scripts/audits/train_visibility_selected_scope_next31.py')
assert registration['visibility_implementation_sha256']==sha256_file('radio_gs/training/selected_output_supervision.py')
assert evaluation['training_registration_sha256']==sha256_file(base/'registration.json')
assert evaluation['evaluator_sha256']==sha256_file('scripts/audits/evaluate_full_field_rebuild_next19.py')
assert evaluation['runner_sha256']==sha256_file('scripts/audits/evaluate_selected_visibility_next31.py')
scenes=[row['scene'] for row in registration['scenes']]
assert scenes==evaluation['scenes']
rows={};bindings={}
def record(path):
    bindings[str(path)]=sha256_file(path)
    return json.loads(path.read_text())
def load(path,digest):
    assert sha256_file(path)==digest
    return torch.load(path,map_location='cpu',weights_only=False)
for row in registration['scenes']:
    scene=row['scene'];root=Path(row['root']);parent=Path(row['parent'])
    parent_report=record(parent/registration['parent_name']/'report.json')
    original=load(parent/registration['parent_name']/'field.pt',row['parent_field_sha256'])
    teachers=record(root/'teachers.json');authority=record(root/'responsibility.json')
    assert teachers['authority_sha256']==sha256_file(root/'responsibility.json')
    caprecord=record(root/'caption_region_evidence.json')
    caption=load(root/'caption_region_evidence.pt',caprecord['sha256'])
    assert caption['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    assert caption['geometry_binding']==original['geometry_binding'] and caption['source_only'] and not caption['benchmark_data_used']
    bindings[str(root/'caption_region_evidence.pt')]=caprecord['sha256']
    baseline=record(parent.parent.parent/'evaluation/report.json')
    def compact(report):
        return dict(lerf2d_iou=report['lerf2d']['final']['miou'],ap=report['lerf2d']['final']['macro_ap'],
            localization=report['lerf2d']['final']['localization_accuracy'],selected_iou=report['selected']['final']['miou'],
            selected_acc25=report['selected']['final']['acc025'],selected_acc50=report['selected']['final']['acc05'])
    rows[scene]=dict(parent=compact(baseline),arms={})
    for mode in evaluation['modes']:
        train=record(root/(mode+'_next31')/'report.json')
        assert train['implementation_sha256']==registration['implementation_sha256']
        assert train['visibility_implementation_sha256']==registration['visibility_implementation_sha256']
        assert train['source_only'] and not train['benchmark_data_used']
        er=record(base/scene/('evaluation_'+mode)/'report.json')
        assert er['complete'] and er['cold_functional_replay_exact']
        assert er['observations']==baseline['observations'] and er['selected']['final']['missing']==0
        assert er['field_sha256']==train['field_sha256'] and train['parent_field_sha256']==row['parent_field_sha256']
        candidate=load(root/(mode+'_next31')/'field.pt',train['field_sha256'])
        allowed=('grounded_transfer.decoder.','grounded_transfer.residual.')
        changed=[]
        for key,value in original['state_dict'].items():
            same=torch.equal(value,candidate['state_dict'][key])
            if not key.startswith(allowed):assert same,key
            if not same:changed.append(key)
        assert original['geometry_binding']==candidate['geometry_binding']
        # Native masks are independent of trained heads, so insist on exact
        # complete-cohort replay in both frozen rendering contracts.
        for sub in ('lerf2d','selected'):
            pp=parent.parent.parent/'evaluation/native_compact_calibrated'/sub
            cp=base/scene/('evaluation_'+mode)/'native_compact_calibrated'/sub
            files=sorted(pp.glob('*.png'));assert len(files)==er['observations']
            assert len(list(cp.glob('*.png')))==len(files)
            for f in files:assert sha256_file(f)==sha256_file(cp/f.name),f
        for kind in ('image','prompt'):
            assert er['reference_development']['macro'][kind]==baseline['reference_development']['macro'][kind]
        rows[scene]['arms'][mode]=dict(metrics=compact(er),selected_step=train['selected_step'],
            changed_parameters=changed,source_initial=train['history'][0]['validation'],
            source_selected=next(h['validation'] for h in train['history'] if h['step']==train['selected_step']),
            native_mask_replay_exact=True,image_prompt_macro_exact=True)
        del candidate
    del caption,original
arms=['parent']+evaluation['modes']
macro={name:{metric:sum((row['parent'] if name=='parent' else row['arms'][name]['metrics'])[metric] for row in rows.values())/4
             for metric in next(iter(rows.values()))['parent']} for name in arms}
result=dict(complete=True,mainline_promoted=False,independent_test=False,scenes=scenes,observations=208,
    method='Fixed geometry and feature tensors; source-only text scope decoder/residual refit with paired selected-silhouette loss.',
    source_budget_changed=False,complete_field_reconstruction_repeated=False,
    inherited_raw_report_metadata_not_applicable=['source_budget_changed=True','Expanded-source complete field reconstruction'],
    source_validation='Held out from the new head fit only; parent saw these source views.',
    physical_3d_truth=False,macro=macro,rows=rows,input_bindings=bindings,
    training_registration_sha256=sha256_file(base/'registration.json'),evaluation_registration_sha256=sha256_file(base/'evaluation_registration.json'),
    implementation_sha256=sha256_file(__file__))
with (base/'summary.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(macro,indent=2))
