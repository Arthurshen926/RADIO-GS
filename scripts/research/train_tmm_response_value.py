"""Controlled set-utility architecture test on the SAME joint supervision."""
import argparse
import io
import json
from pathlib import Path
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.response_replacement import FEATURE_NAMES, calibrate_margin
from radio_gs.querying.response_supervision import calibration_groups
from radio_gs.querying.response_value import training_values, relative_utilities
from scripts.research.train_tmm_joint_replacement import read, load_reports, evaluate
from scripts.research.replay_tmm_response import STUDY, POLICY, sha


def predict(reports, model, *, blind=False, permuted=False):
    return [dict(scene=r['scene'], modality=r['modality'], query_index=row['query_index'], ious=row['ious'],
                 gains=relative_utilities(model, row['permuted_features' if permuted else 'features'], query_blind=blind).tolist(),
                 evidence_available=row['evidence']['verification_views'] > 0)
            for r in reports for row in r['rows']]


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', default='response_value_train_v1'); args = p.parse_args()
    protocol_path = Path('configs/research/response_value.json'); protocol = read(protocol_path)
    config = read(protocol['parent_supervision_config']); parent = STUDY/'response_replacement_joint_v1'
    original = read(parent/'manifest.json'); splits = original['splits']
    if config != original['config'] or protocol['image_run'] != original['image_run']:
        raise ValueError('controlled utility test requires identical joint supervision')
    if any(sha(p) != h for p, h in original['input_hashes'].items()):
        raise ValueError('sealed supervision changed')
    config = dict(config, value_protocol=protocol)
    paths = [str(protocol_path), protocol['parent_supervision_config'], __file__,
        'radio_gs/querying/response_value.py', 'radio_gs/querying/response_replacement.py',
        'radio_gs/querying/response_supervision.py', 'scripts/research/train_tmm_joint_replacement.py',
        'scripts/research/train_tmm_response_replacement.py']
    run = BoundedExperiment(STUDY, args.run, POLICY); run.preflight(8*2**20)
    manifest = dict(config=config, splits=splits, image_run=original['image_run'],
        source_hashes={s: sha(s) for s in paths}, input_hashes=original['input_hashes'],
        controlled_parent_manifest_sha256=sha(parent/'manifest.json'), decision_rule='set_value',
        benchmark_feedback_used=False, original_joint_calibration_feedback_used=True)
    run.json('manifest.json', manifest); run.json('source_snapshot.json', {s: Path(s).read_text() for s in paths})
    text = STUDY/config['text_run']; images = STUDY/protocol['image_run']
    fit = load_reports(text, images, splits['fit']); cal_reports = load_reports(text, images, splits['calibration'])
    bundle = dict(feature_names=FEATURE_NAMES, config=config, splits=splits, models={}, margins={}, decision_rule='set_value',
        learned_query_types='independent natural text and real held-out RGB queries with SigLIP + DINO',
        calibration_incumbent='text scope and image direct; one margin across six room/modality cells',
        training_manifest_sha256=sha(run.root/'manifest.json'))
    calibration = {}
    for condition in config['conditions']:
        blind = condition == 'query_blind'; x, y, w = training_values(fit, query_blind=blind)
        model = ExtraTreesRegressor(**config['estimator'], n_jobs=4).fit(x, y, sample_weight=w); model.n_jobs = 1
        rows = predict(cal_reports, model, blind=blind); selected, detail = calibrate_margin(calibration_groups(rows))
        bundle['models'][condition] = model; bundle['margins'][condition] = selected['margin']
        calibration[condition] = dict(selection=selected, rule=detail, evaluation=evaluate(rows, selected['margin']),
            training_candidates=len(y), training_queries=sum(r['queries'] for r in fit))
        print('set utility fit/calibrated', condition, selected, flush=True)
    buffer = io.BytesIO(); joblib.dump(bundle, buffer, compress=3); payload = buffer.getvalue()
    run._write('model.joblib', lambda f: f.write(payload), limit=len(payload)); run.json('calibration.json', calibration)
    run.json('model_seal.json', dict(model_sha256=sha(run.root/'model.joblib'), calibration_sha256=sha(run.root/'calibration.json'),
        development_feedback_used=False, benchmark_feedback_used=False, default_changed=False,
        previous_joint_calibration_feedback_used=True))
    development = load_reports(text, images, splits['validation']); results = {}
    for condition in config['conditions']:
        results[condition] = evaluate(predict(development, bundle['models'][condition], blind=condition == 'query_blind'), bundle['margins'][condition])
    results['permuted'] = evaluate(predict(development, bundle['models']['full'], permuted=True), bundle['margins']['full'])
    results['unrestricted'] = evaluate(predict(development, bundle['models']['full']), 0.)
    gates = {}
    for m in ('text', 'image'):
        r = results['full']['by_modality'][m]
        gates[m] = (r['scene_macro']['delta'] > 0
            and all(r['scene_macro']['selected_iou'] > results[a]['by_modality'][m]['scene_macro']['selected_iou'] for a in ('query_blind', 'permuted'))
            and sum(s['delta'] >= 0 for s in r['per_scene'].values()) >= 3)
    summary = dict(complete=True, decision_rule='set_value', development=results, development_gate=all(gates.values()), modality_gates=gates,
        training_queries={m: sum(r['queries'] for r in fit if r['modality'] == m) for m in ('text', 'image')},
        calibration_queries={m: sum(r['queries'] for r in cal_reports if r['modality'] == m) for m in ('text', 'image')},
        development_queries={m: sum(r['queries'] for r in development if r['modality'] == m) for m in ('text', 'image')},
        checkpoint_sha256=sha(run.root/'model.joblib'), independent_final_test=False, default_changed=False,
        all_five_transfer_pending=True, durable_dense_prediction_bytes=0, study_bytes=tree_bytes(STUDY))
    run.json('summary.json', summary)
    print(json.dumps({a: {m: r['scene_macro'] for m, r in v['by_modality'].items()} for a,v in results.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
