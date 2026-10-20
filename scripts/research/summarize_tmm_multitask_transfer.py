"""Keep measured transfer, ready candidates, and missing outputs separate."""
import json
from pathlib import Path
from radio_gs.response_study.registry import read,sha


def main():
    registry=read('configs/research/tmm_response.json');study=Path(registry['study_root'])
    reference=read(registry['reference_summary'])['macro'];prepared=study/'category_transfer_inputs_v2'
    training=study/'response_value_multitask_v1';transfer=study/'response_value_multitask_transfer_v1'
    diagnosis=study/'response_value_multitask_diagnosis_v1'
    summary=read(transfer/'summary.json') if (transfer/'summary.json').exists() else None
    specs=[('lerf_semantic2d','lerf',['semantic_iou','ap']),
        ('lerf_selected_set_projection2d','lerf',['selected_projection_iou']),
        ('nvos','nvos',list(reference['nvos'])),('scannet_ovs','scannet',list(reference['scannet'])),
        ('uqis','uqis',list(reference['uqis']))]
    outputs=[]
    for name,family,keys in specs:
        measured=summary is not None and family in ('scannet','uqis')
        arms=(summary['category_macro'] if family=='scannet' else summary['uqis_macro']) if measured else {}
        outputs.append(dict(output=name,reference_percent={k:100*reference[family][k] for k in keys},
            status='freshly_scored' if measured else 'transfer_pending',
            values_percent={a:{k:100*v[k] for k in keys} for a,v in arms.items()},
            delta_pp={a:{k:100*(v[k]-reference[family][k]) for k in keys} for a,v in arms.items() if a!='baseline'},
            intervention_scope=('text/image; original point answers exactly retained' if family=='uqis'
                else 'one complete 19-class menu shared across submenus' if family=='scannet' else 'not yet implemented')))
    feedback=dict(outputs=outputs,default_changed=False,independent_final_test=False,
        all_five_transfer_complete=False,category_pools_ready=(prepared/'summary.json').exists(),
        model_fitted=(training/'summary.json').exists(),fresh_joint_transfer_complete=summary is not None,
        stages={name:read(study/name/'progress.json') for name in
            ('category_multitask_program_v1','multitask_transfer_program_v2') if (study/name/'progress.json').exists()},
        source_rooms_complete=len(list((study/'category_utility_supervision_v1').glob('scene*.json'))),
        source_rooms_required=29,mechanism='one set-conditioned utility and one calibrated margin; intact answers only',
        limitations=['opened benchmark/development data; no untouched final test',
            'mesh versus Gaussian carrier and full versus region-covered visibility domains differ',
            'six category calibration classes absent; AP and independent LERF set not supervised'],
        receipts={str(p):sha(p) for p in (prepared/'summary.json',training/'summary.json',transfer/'summary.json',diagnosis/'summary.json') if p.exists()})
    if (prepared/'summary.json').exists():
        prep=read(prepared/'summary.json'); rows=[read(prepared/n) for n in prep['report_hashes']]
        feedback['prepared']=dict(scenes=len(rows),views=sum(r['views'] for r in rows),regions=sum(r['regions'] for r in rows),
            maximum_replay_error=max(r['compatibility'][k]['max_abs'] for r in rows for k in ('primary','fusion')),
            changed_original_labels=sum(sum(r['compatibility'][k]['changed_subset_labels'].values()) for r in rows for k in ('primary','fusion')),
            volatile_prediction_bytes=prep['transient_prediction_bytes'],durable_dense_prediction_bytes=0,
            prediction_bank_currently_available=Path(read(prepared/'manifest.json')['ram_directory']).is_dir(),
            release_receipt_exists=(transfer/'ram_released.json').exists())
    if (diagnosis/'summary.json').exists():
        data=read(diagnosis/'summary.json')
        feedback['diagnosis']={k:data[k] for k in ('category_macro','instance_macro','category_same_choice_as_blind','category_same_choice_as_permuted')}
    Path('paper/tmm_response/multitask_transfer_feedback.json').write_text(json.dumps(feedback,indent=2)+'\n')
    print(json.dumps(feedback,indent=2))


if __name__=='__main__':main()
