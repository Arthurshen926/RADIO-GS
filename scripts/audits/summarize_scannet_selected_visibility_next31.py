"""Summarize the complete frozen eight-scene paired scope-refit experiment."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


root=Path('output/scannet_selected_visibility_next31')
registration=json.loads((root/'registration.json').read_text())
rows=[]
for item in registration['scenes']:
    scene=item['scene'];path=root/scene/'evaluation/report.json'
    report=json.loads(path.read_text())
    assert all(v==0 for by_split in report['baseline_mismatches'].values() for v in by_split.values())
    source={}
    for mode in registration['modes']:
        receipt=root/scene/mode/'report.json';r=json.loads(receipt.read_text())
        assert r['parent_field_sha256']==item['parent_field_sha256']
        assert sha256_file(receipt)==report['source_refit_receipts'][mode]['report_sha256']
        initial=r['history'][0]['validation'];chosen=next(x['validation'] for x in r['history'] if x['step']==r['selected_step'])
        assert chosen['full_iou']>=initial['full_iou']-1e-6 and (r['selected_step']==0 or chosen['selected_iou']>initial['selected_iou']+1e-6)
        source[mode]=dict(selected_step=r['selected_step'],initial=initial,chosen=chosen,
            camera_replay_max_error=r['camera_replay_max_error'],report_sha256=sha256_file(receipt))
    rows.append(dict(scene=scene,metrics=report['metrics'],decisions=report['decisions'],source=source,
        evaluation_report_sha256=sha256_file(path)))
assert len(rows)==8
arms=['native','retained','selected','visible_selected']
macro={arm:{split:sum(r['metrics'][arm][split]['miou'] for r in rows)/8 for split in ('19','15','10')} for arm in arms}
source_macro={mode:{kind:{metric:sum(r['source'][mode][kind][metric] for r in rows)/8
    for metric in ('full_iou','selected_iou')} for kind in ('initial','chosen')} for mode in registration['modes']}
summary=dict(schema='scannet_selected_visibility_complete.next31',complete=True,scenes=8,
    paired_source_refits=16,steps_per_refit=400,source_only_fit=True,independent_test=False,
    input_track='provided_unlabelled_mesh',frozen_protocol='scannet-ovs-paper8-v1',macro=macro,
    delta_pp={arm:{split:100*(macro[arm][split]-macro['retained'][split]) for split in ('19','15','10')}
        for arm in registration['modes']},source_macro=source_macro,rows=rows,
    cold_native_and_retained_prediction_replay_exact=True,original_native_coordinates_exact=True,
    original_source_scope_acceptance_retained=True,no_target_scene_branch_selection=True,
    inherited_rejected_scenes=[r['scene'] for r in rows if not r['decisions']['retained']['use_scope']],
    decision='Keep unchanged retained baseline; source binary-scope gains alone do not establish category retrieval improvement.',
    registration_sha256=sha256_file(root/'registration.json'),evaluation_registration_sha256=sha256_file(root/'evaluation_registration.json'),
    implementation_sha256=sha256_file(__file__))
with (root/'summary.json').open('x') as handle:json.dump(summary,handle,indent=2)
print(json.dumps(dict(macro=macro,delta_pp=summary['delta_pp'],source_macro=source_macro),indent=2))
