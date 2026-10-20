"""Fit one residual-utility hypothesis and its matched no-null control."""
import argparse
import io
from pathlib import Path
import joblib
from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.residual_response_value import fit_residual_models
from radio_gs.querying.response_replacement import FEATURE_NAMES, calibrate_margin
from radio_gs.querying.multitask_response_value import predict, protected_output_rows
from radio_gs.querying.response_study_adapters import read
from scripts.research.train_tmm_multitask_value import load_all, evaluate, TASKS
from scripts.research.replay_tmm_response import STUDY, POLICY, sha

CONFIG = Path('configs/research/response_value_residual.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='response_value_residual_v1'); args = parser.parse_args()
    design = read(CONFIG); parent = STUDY/design['parent_training_run']
    manifest = read(parent/'manifest.json'); previous = read(parent/'summary.json')
    if not previous['complete'] or design['null_weight'] != .5:
        raise ValueError('fixed completed joint supervision and matched half-weight nulls required')
    for path, digest in {**manifest['input_hashes'], **manifest['source_hashes']}.items():
        if sha(path) != digest:
            raise ValueError('parent supervision or implementation changed: '+path)
    sources = [str(CONFIG), __file__, 'radio_gs/querying/residual_response_value.py',
        'radio_gs/querying/response_value.py', 'radio_gs/querying/response_replacement.py',
        'radio_gs/querying/multitask_response_value.py', 'scripts/research/train_tmm_multitask_value.py',
        'scripts/research/train_tmm_joint_replacement.py', 'scripts/research/train_tmm_response_replacement.py']
    hashes = {p:sha(p) for p in sources}; inputs = dict(manifest['input_hashes'])
    inputs[str(parent/'manifest.json')] = sha(parent/'manifest.json')
    config = dict(manifest['config'], utility_architecture=design['method']); splits=manifest['splits']
    run = BoundedExperiment(STUDY,args.run,POLICY); run.preflight(4*2**20)
    run.json('manifest.json', dict(config=config, design=design, splits=splits, estimator=manifest['estimator'],
        input_hashes=inputs, source_hashes=hashes, decision_rule='set_value',
        benchmark_parameters_fitted=False, previous_benchmark_feedback_used_for_design=True,
        prior_calibration_feedback_used=True))
    run.json('source_snapshot.json', {p:Path(p).read_text() for p in sources})
    fit = load_all(config,splits['fit'],fit=True)
    models, detail = fit_residual_models(fit,manifest['estimator'],design['room_folds'])
    run.json('fit_diagnostics.json',detail)
    cal = load_all(config,splits['calibration']); calibration={}; margins={}
    for arm, model in models.items():
        rows = predict(cal,model,blind=arm=='query_blind')
        selected, rules = calibrate_margin(protected_output_rows(rows))
        if len(selected['per_scene']) != 15:
            raise ValueError('all fifteen protected calibration cells required')
        margins[arm] = selected['margin']
        calibration[arm] = dict(selection=selected, rule=rules, evaluation=evaluate(rows,selected['margin']))
        print('calibration',arm,selected,flush=True)
    bundle = dict(feature_names=FEATURE_NAMES,config=config,splits=splits,models=models,margins=margins,
        decision_rule='set_value',training_manifest_sha256=sha(run.root/'manifest.json'))
    buffer=io.BytesIO(); joblib.dump(bundle,buffer,compress=3); payload=buffer.getvalue()
    run._write('model.joblib',lambda f:f.write(payload),limit=len(payload))
    run.json('calibration.json',calibration)
    run.json('model_seal.json',dict(model_sha256=sha(run.root/'model.joblib'),calibration_sha256=sha(run.root/'calibration.json'),
        development_feedback_used=False,benchmark_feedback_used=False,default_changed=False))
    dev=load_all(config,splits['validation']); results={}
    for arm in ('full','query_blind','permuted','unrestricted','unregularized'):
        key=arm if arm in models else 'full'; margin=0. if arm=='unrestricted' else margins[key]
        results[arm]=evaluate(predict(dev,models[key],blind=key=='query_blind',permuted=arm=='permuted'),margin)
    gates={m: (results['full']['by_modality'][m]['scene_macro']['delta']>0
        and sum(v['delta']>=0 for v in results['full']['by_modality'][m]['per_scene'].values())>=3
        and all(results['full']['by_modality'][m]['scene_macro']['selected_iou']>
                results[c]['by_modality'][m]['scene_macro']['selected_iou'] for c in ('query_blind','permuted')))
        for m in TASKS}
    protected=all(v['scene_macro']['delta']>=0 for v in results['full']['category_outputs'].values())
    if any(sha(p)!=h for p,h in {**inputs,**hashes}.items()):
        raise ValueError('fixed training inputs or source changed')
    run.json('summary.json',dict(complete=True,development=results,modality_gates=gates,
        development_gate=all(gates.values()) and protected,all_menu_outputs_nonnegative=protected,
        training_queries={m:sum(r['queries'] for r in fit if r['modality']==m) for m in TASKS},
        calibration_queries={m:sum(r['queries'] for r in cal if r['modality']==m) for m in TASKS},
        development_queries={m:sum(r['queries'] for r in dev if r['modality']==m) for m in TASKS},
        checkpoint_sha256=sha(run.root/'model.joblib'),independent_final_test=False,default_changed=False,
        all_five_transfer_pending=True,calibration_gap_retained=previous['calibration_gap_retained'],
        durable_dense_prediction_bytes=0,study_bytes=tree_bytes(STUDY),fit_diagnostics_sha256=sha(run.root/'fit_diagnostics.json')))
    print('development', {a:{m:v['scene_macro'] for m,v in r['by_modality'].items()} for a,r in results.items()},flush=True)


if __name__=='__main__':main()
