"""One fixed architecture control on the exact completed joint supervision."""
import argparse
import io
from pathlib import Path
import time
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from radio_gs.query_field.bounded_experiment import BoundedExperiment,tree_bytes
from radio_gs.querying.centered_response_value import CenteredResponseValue
from radio_gs.querying.response_replacement import FEATURE_NAMES,calibrate_margin
from radio_gs.querying.response_value import training_values
from radio_gs.querying.multitask_response_value import predict,protected_output_rows
from radio_gs.querying.response_study_adapters import read
from scripts.research.train_tmm_multitask_value import load_all,evaluate,TASKS
from scripts.research.replay_tmm_response import STUDY,POLICY,sha

CONFIG=Path('configs/research/response_value_centered.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='response_value_centered_v1')
    p.add_argument('--wait-for-parent',action='store_true');args=p.parse_args()
    design=read(CONFIG);parent=STUDY/design['parent_training_run']
    # Freeze the control before waiting for any parent result.
    sources=[str(CONFIG),__file__,'radio_gs/querying/centered_response_value.py',
        'radio_gs/querying/response_value.py','radio_gs/querying/multitask_response_value.py',
        'radio_gs/querying/response_replacement.py','scripts/research/train_tmm_multitask_value.py',
        'scripts/research/train_tmm_joint_replacement.py','scripts/research/train_tmm_response_replacement.py']
    hashes={s:sha(s) for s in sources}
    run=BoundedExperiment(STUDY,args.run,POLICY);run.preflight(6*2**20)
    run.json('design_seal.json',dict(design=design,source_hashes=hashes,parent_complete_at_design=(parent/'summary.json').exists(),
        parent_benchmark_complete_at_design=(STUDY/'response_value_multitask_transfer_v1/summary.json').exists()))
    run.json('source_snapshot.json',{s:Path(s).read_text() for s in sources})
    while args.wait_for_parent and not (parent/'summary.json').exists():
        if read(STUDY/'category_multitask_program_v1/progress.json')['status']=='failed':
            raise RuntimeError('parent corpus/training failed')
        print('centered utility control waiting for complete parent training',flush=True);time.sleep(30)
    previous=read(parent/'summary.json');manifest=read(parent/'manifest.json')
    if not previous['complete']:raise ValueError('complete parent training required')
    for path,digest in {**manifest['input_hashes'],**manifest['source_hashes'],**hashes}.items():
        if sha(path)!=digest:raise ValueError('fixed input or implementation changed: '+path)
    config=dict(manifest['config'],utility_context='candidate-minus-set-mean only')
    splits=manifest['splits'];inputs=dict(manifest['input_hashes'])
    inputs[str(parent/'manifest.json')]=sha(parent/'manifest.json')
    inputs[str(parent/'summary.json')]=sha(parent/'summary.json')
    run.json('manifest.json',dict(config=config,design=design,splits=splits,estimator=manifest['estimator'],
        source_hashes=hashes,input_hashes=inputs,decision_rule='set_value',benchmark_feedback_used=False,
        prior_calibration_feedback_used=True,parent_training_run=design['parent_training_run']))
    fit=load_all(config,splits['fit'],fit=True);cal=load_all(config,splits['calibration'])
    bundle=dict(feature_names=FEATURE_NAMES,config=config,splits=splits,models={},margins={},decision_rule='set_value',
        training_manifest_sha256=sha(run.root/'manifest.json'),category_gaussian_transfer_pending=True)
    calibration={}
    for condition in ('full','query_blind'):
        blind=condition=='query_blind';x,y,w=training_values(fit,query_blind=blind)
        base=ExtraTreesRegressor(**manifest['estimator'],n_jobs=4)
        model=CenteredResponseValue(base).fit(x,y,sample_weight=w);base.n_jobs=1
        rows=predict(cal,model,blind=blind);selected,detail=calibrate_margin(protected_output_rows(rows))
        if len(selected['per_scene'])!=15:raise ValueError('same fifteen calibration cells required')
        bundle['models'][condition]=model;bundle['margins'][condition]=selected['margin']
        calibration[condition]=dict(selection=selected,rule=detail,evaluation=evaluate(rows,selected['margin']))
        print('centered utility fit',condition,selected,flush=True)
    buffer=io.BytesIO();joblib.dump(bundle,buffer,compress=3);payload=buffer.getvalue()
    run._write('model.joblib',lambda f:f.write(payload),limit=len(payload));run.json('calibration.json',calibration)
    run.json('model_seal.json',dict(model_sha256=sha(run.root/'model.joblib'),calibration_sha256=sha(run.root/'calibration.json'),
        development_feedback_used=False,benchmark_feedback_used=False,default_changed=False))
    dev=load_all(config,splits['validation']);results={}
    for arm in ('full','query_blind','permuted','unrestricted'):
        key='query_blind' if arm=='query_blind' else 'full';margin=0. if arm=='unrestricted' else bundle['margins'][key]
        results[arm]=evaluate(predict(dev,bundle['models'][key],blind=key=='query_blind',permuted=arm=='permuted'),margin)
    gates={}
    for m in TASKS:
        r=results['full']['by_modality'][m]
        gates[m]=(r['scene_macro']['delta']>0 and sum(s['delta']>=0 for s in r['per_scene'].values())>=3
            and all(r['scene_macro']['selected_iou']>results[c]['by_modality'][m]['scene_macro']['selected_iou'] for c in ('query_blind','permuted')))
    menu_nonnegative=all(v['scene_macro']['delta']>=0 for v in results['full']['category_outputs'].values())
    if any(sha(p)!=h for p,h in inputs.items()):raise ValueError('supervision changed during control fit')
    run.json('summary.json',dict(complete=True,development=results,modality_gates=gates,
        development_gate=all(gates.values()) and menu_nonnegative,all_menu_outputs_nonnegative=menu_nonnegative,
        training_queries={m:sum(r['queries'] for r in fit if r['modality']==m) for m in TASKS},
        calibration_queries={m:sum(r['queries'] for r in cal if r['modality']==m) for m in TASKS},
        development_queries={m:sum(r['queries'] for r in dev if r['modality']==m) for m in TASKS},
        checkpoint_sha256=sha(run.root/'model.joblib'),independent_final_test=False,default_changed=False,
        all_five_transfer_pending=True,calibration_gap_retained=previous['calibration_gap_retained'],
        durable_dense_prediction_bytes=0,study_bytes=tree_bytes(STUDY),
        single_changed_factor='remove absolute candidate and candidate-set mean context; keep centered signature only'))
    print('centered control development', {a:{m:r['scene_macro'] for m,r in v['by_modality'].items()} for a,v in results.items()},flush=True)


if __name__=='__main__':main()
