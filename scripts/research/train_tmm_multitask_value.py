"""Fit one complete-answer utility on text, images and full category menus.

All 29 full-teacher category reports must finish before fitting. One global
margin protects 15 room/output cells. No benchmark is read or tuned here.
"""
import argparse
import io
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.response_replacement import FEATURE_NAMES, calibrate_margin
from radio_gs.querying.response_value import training_values
from radio_gs.querying.multitask_response_value import category_report, predict, protected_output_rows
from scripts.research.train_tmm_joint_replacement import load_reports, read
from scripts.research.train_tmm_response_replacement import summarize
from scripts.research.replay_tmm_response import STUDY, POLICY, sha

CONFIG = Path('configs/research/response_value_multitask.json')
TASKS = ('text', 'image', 'category')


def load_all(config, scenes, *, fit=False):
    reports = load_reports(STUDY/config['text_run'], STUDY/config['image_run'], scenes)
    reports += [category_report(read(STUDY/config['category_run']/(s+'.json')), fit=fit) for s in scenes]
    if any(sum(r['scene'] == s and r['modality'] == m for r in reports) != 1 for s in scenes for m in TASKS):
        raise ValueError('every room/task must have exactly one balanced supervision group')
    return reports


def evaluate(rows, margin):
    tasks = {m: summarize([r for r in rows if r['modality'] == m], margin) for m in TASKS}
    menu = [r for r in rows if r['modality'] == 'category']
    outputs = {k: summarize([dict(r, ious=r['menu_utilities'][k]) for r in menu], margin) for k in ('19','15','10')}
    choices = [[r['candidate'] for r in v['records']] for v in outputs.values()]
    if any(c != choices[0] for c in choices):
        raise AssertionError('submenus must share a complete-answer decision')
    return dict(by_modality=tasks, category_outputs=outputs, same_menu_decisions=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', default='response_value_multitask_v1')
    args = p.parse_args(); protocol = read(CONFIG); parent = read(protocol['parent_config'])
    inherited = read(STUDY/'response_replacement_joint_v1/manifest.json')
    if any(sha(p) != h for p,h in inherited['input_hashes'].items()):
        raise ValueError('previously sealed text/image supervision changed')
    category = STUDY/protocol['category_run']
    summary = read(category/'summary.json'); manifest = read(category/'manifest.json')
    if not summary['complete'] or not summary['native_and_source_supervision_ready'] or summary['rooms'] != 29:
        raise ValueError('all original-teacher category episodes must be complete before fitting')
    splits = manifest['splits']; paths = []
    for key in ('text_run','image_run','category_run'):
        root = STUDY/protocol[key]; data = read(root/'manifest.json')
        if data['splits'] != splits or not read(root/'summary.json')['complete']:
            raise ValueError('all tasks require the same complete room split')
        paths += [root/'manifest.json', root/'summary.json']
        paths += [root/(s+'.json') for s in sum(splits.values(), [])]
    for name, digest in summary['report_hashes'].items():
        if sha(category/name) != digest:
            raise ValueError('category supervision changed after compilation')
    sources = [str(CONFIG), protocol['parent_config'], __file__, 'radio_gs/querying/multitask_response_value.py',
        'radio_gs/querying/response_value.py', 'radio_gs/querying/response_replacement.py',
        'scripts/research/train_tmm_joint_replacement.py', 'scripts/research/train_tmm_response_replacement.py',
        'radio_gs/querying/response_supervision.py']
    inputs = {str(p): sha(p) for p in paths}
    run = BoundedExperiment(STUDY, args.run, POLICY); run.preflight(6*2**20)
    run.json('manifest.json', dict(config=protocol, splits=splits, estimator=parent['estimator'],
        source_hashes={s: sha(s) for s in sources}, input_hashes=inputs,
        decision_rule='set_value', benchmark_feedback_used=False, prior_calibration_feedback_used=True))
    run.json('source_snapshot.json', {s: Path(s).read_text() for s in sources})
    fit = load_all(protocol, splits['fit'], fit=True); cal = load_all(protocol, splits['calibration'])
    bundle = dict(feature_names=FEATURE_NAMES, config=protocol, splits=splits, models={}, margins={}, decision_rule='set_value',
        training_manifest_sha256=sha(run.root/'manifest.json'), category_gaussian_transfer_pending=True)
    calibration = {}
    for condition in parent['conditions']:
        blind = condition == 'query_blind'; x, y, w = training_values(fit, query_blind=blind)
        model = ExtraTreesRegressor(**parent['estimator'], n_jobs=4).fit(x,y,sample_weight=w); model.n_jobs=1
        rows = predict(cal, model, blind=blind)
        selected, detail = calibrate_margin(protected_output_rows(rows))
        if len(selected['per_scene']) != 15:
            raise AssertionError('three rooms and five protected outputs required')
        bundle['models'][condition] = model; bundle['margins'][condition] = selected['margin']
        calibration[condition] = dict(selection=selected, rule=detail, evaluation=evaluate(rows,selected['margin']))
        print('multitask fit', condition, selected, flush=True)
    buffer=io.BytesIO(); joblib.dump(bundle,buffer,compress=3); payload=buffer.getvalue()
    run._write('model.joblib',lambda f:f.write(payload),limit=len(payload)); run.json('calibration.json',calibration)
    run.json('model_seal.json',dict(model_sha256=sha(run.root/'model.joblib'),calibration_sha256=sha(run.root/'calibration.json'),
        development_feedback_used=False, benchmark_feedback_used=False, default_changed=False))
    dev=load_all(protocol,splits['validation']); results={}
    for condition in parent['conditions']:
        results[condition]=evaluate(predict(dev,bundle['models'][condition],blind=condition=='query_blind'),bundle['margins'][condition])
    results['permuted']=evaluate(predict(dev,bundle['models']['full'],permuted=True),bundle['margins']['full'])
    results['unrestricted']=evaluate(predict(dev,bundle['models']['full']),0.)
    gates={}
    for m in TASKS:
        r=results['full']['by_modality'][m]
        gates[m]=(r['scene_macro']['delta']>0 and sum(s['delta']>=0 for s in r['per_scene'].values())>=3
            and all(r['scene_macro']['selected_iou']>results[c]['by_modality'][m]['scene_macro']['selected_iou'] for c in ('query_blind','permuted')))
    menu_nonnegative=all(v['scene_macro']['delta']>=0 for v in results['full']['category_outputs'].values())
    if any(sha(p)!=h for p,h in inputs.items()):
        raise ValueError('supervision changed while fitting')
    run.json('summary.json',dict(complete=True,development=results,modality_gates=gates,
        development_gate=all(gates.values()) and menu_nonnegative, all_menu_outputs_nonnegative=menu_nonnegative,
        training_queries={m:sum(r['queries'] for r in fit if r['modality']==m) for m in TASKS},
        calibration_queries={m:sum(r['queries'] for r in cal if r['modality']==m) for m in TASKS},
        development_queries={m:sum(r['queries'] for r in dev if r['modality']==m) for m in TASKS},
        checkpoint_sha256=sha(run.root/'model.joblib'), independent_final_test=False, default_changed=False,
        all_five_transfer_pending=True, calibration_gap_retained=summary['calibration_gap_retained'],
        durable_dense_prediction_bytes=0, study_bytes=tree_bytes(STUDY)))
    print(json.dumps({a:{m:r['scene_macro'] for m,r in v['by_modality'].items()} for a,v in results.items()},indent=2))


if __name__ == '__main__':
    main()
