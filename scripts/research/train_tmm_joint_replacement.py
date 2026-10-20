"""One response decision model and margin jointly supervised by text and images."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.response_replacement import FEATURE_NAMES, calibrate_margin
from radio_gs.querying.response_supervision import incumbent_first, calibration_groups
from scripts.research.train_tmm_response_replacement import training_pairs, infer_rows, summarize
from scripts.research.replay_tmm_response import STUDY, POLICY, sha

CONFIG = Path('configs/research/response_replacement_joint.json')


def read(path):
    return json.loads(Path(path).read_text())


def load_reports(text_root, image_root, scenes):
    reports = []
    for scene in scenes:
        for modality, root, provider in [('text', text_root, 'scope'), ('image', image_root, 'direct')]:
            report = read(root/(scene+'.json'))
            rows = [dict(incumbent_first(r, provider), modality=modality) for r in report['rows']]
            if not rows:
                raise ValueError(f'empty required supervision cell: {scene}/{modality}')
            reports.append(dict(report, rows=rows, modality=modality))
    return reports


def predict(reports, model, *, blind=False, permuted=False):
    rows = infer_rows(reports, model, blind=blind, permuted=permuted)
    modalities = [report['modality'] for report in reports for _ in report['rows']]
    return [dict(row, modality=modality) for row, modality in zip(rows, modalities)]


def evaluate(rows, margin):
    by_modality = {m: summarize([r for r in rows if r['modality'] == m], margin) for m in ('text', 'image')}
    return dict(by_modality=by_modality,
        balanced_macro={k: float(np.mean([r['scene_macro'][k] for r in by_modality.values()]))
                        for k in by_modality['text']['scene_macro']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', default='response_replacement_images_v2')
    p.add_argument('--run', default='response_replacement_joint_v1')
    args = p.parse_args(); config = read(CONFIG)
    text_root = STUDY/config['text_run']; image_root = STUDY/args.images
    if not read(image_root/'summary.json')['complete'] or not read(text_root/'summary.json')['complete']:
        raise ValueError('both independent modality tables must be complete')
    text_manifest = read(text_root/'manifest.json'); image_manifest = read(image_root/'manifest.json')
    splits = text_manifest['splits']
    if splits != image_manifest['splits'] or config != image_manifest['config']:
        raise ValueError('joint protocol or split changed since episode compilation')
    for manifest in (text_manifest, image_manifest):
        if any(sha(p) != h for p, h in manifest['source_hashes'].items()):
            raise ValueError('compiled supervision implementation changed')
    paths = [text_root/'manifest.json', text_root/'summary.json', image_root/'manifest.json',
             image_root/'summary.json', image_root/'teacher_seal.json']
    # Development report targets are not read until AFTER fitting and sealing.
    for scene in sum(splits.values(), []):
        paths += [text_root/(scene+'.json'), image_root/(scene+'.json')]
    sources = [str(CONFIG), __file__, 'scripts/research/train_tmm_response_replacement.py',
               'radio_gs/querying/response_replacement.py', 'radio_gs/querying/response_supervision.py']
    input_hashes = {str(path): sha(path) for path in paths}
    input_hashes[config['provider_checkpoint']] = sha(config['provider_checkpoint'])
    run = BoundedExperiment(STUDY, args.run, POLICY); run.preflight(8*2**20)
    run.json('manifest.json', dict(config=config, splits=splits, source_hashes={s: sha(s) for s in sources},
        input_hashes=input_hashes, modalities=['text', 'image'], image_run=args.images,
        supervision='independent natural text + real held-out-view RGB crops, original SigLIP and DINO teachers',
        inherited_decoder_fit_overlap=True, benchmark_labels_used=False))
    run.json('source_snapshot.json', {s: Path(s).read_text() for s in sources})
    fit = load_reports(text_root, image_root, splits['fit'])
    calibration = load_reports(text_root, image_root, splits['calibration'])
    bundle = dict(feature_names=FEATURE_NAMES, models={}, margins={}, config=config, splits=splits,
        training_manifest_sha256=sha(run.root/'manifest.json'),
        learned_query_types='independent natural text and real held-out-view RGB queries with SigLIP + DINO',
        calibration_incumbent='text scope and image direct; one margin across all six room/modality cells')
    cal = {}
    for condition in config['conditions']:
        blind = condition == 'query_blind'
        x, y, weights = training_pairs(fit, blind)
        estimator = ExtraTreesRegressor(**config['estimator'], n_jobs=4).fit(x, y, sample_weight=weights)
        estimator.n_jobs = 1
        rows = predict(calibration, estimator, blind=blind)
        selected, detail = calibrate_margin(calibration_groups(rows))
        bundle['models'][condition] = estimator; bundle['margins'][condition] = selected['margin']
        cal[condition] = dict(selection=selected, rule=detail, evaluation=evaluate(rows, selected['margin']),
                              training_pairs=len(y), training_queries=sum(r['queries'] for r in fit))
        print('joint fit/calibrated', condition, selected, flush=True)
    buffer = io.BytesIO(); joblib.dump(bundle, buffer, compress=3); payload = buffer.getvalue()
    run._write('model.joblib', lambda f: f.write(payload), limit=len(payload)); run.json('calibration.json', cal)
    run.json('model_seal.json', dict(model_sha256=sha(run.root/'model.joblib'),
        calibration_sha256=sha(run.root/'calibration.json'), benchmark_feedback_used=False,
        development_feedback_used=False, default_changed=False))
    development = load_reports(text_root, image_root, splits['validation'])
    result = {}
    for condition in config['conditions']:
        result[condition] = evaluate(predict(development, bundle['models'][condition], blind=condition == 'query_blind'), bundle['margins'][condition])
    result['permuted'] = evaluate(predict(development, bundle['models']['full'], permuted=True), bundle['margins']['full'])
    result['unrestricted'] = evaluate(predict(development, bundle['models']['full']), 0.)
    gates = {}
    for m in ('text', 'image'):
        r = result['full']['by_modality'][m]
        gates[m] = (r['scene_macro']['delta'] > 0
            and all(r['scene_macro']['selected_iou'] > result[a]['by_modality'][m]['scene_macro']['selected_iou'] for a in ('query_blind', 'permuted'))
            and sum(s['delta'] >= 0 for s in r['per_scene'].values()) >= 3)
    if any(sha(p) != h for p, h in input_hashes.items()):
        raise ValueError('supervision inputs changed while fitting')
    summary = dict(complete=True, development=result, development_gate=all(gates.values()), modality_gates=gates,
        training_queries={m: sum(r['queries'] for r in fit if r['modality'] == m) for m in ('text', 'image')},
        calibration_queries={m: sum(r['queries'] for r in calibration if r['modality'] == m) for m in ('text', 'image')},
        development_queries={m: sum(r['queries'] for r in development if r['modality'] == m) for m in ('text', 'image')},
        checkpoint_sha256=sha(run.root/'model.joblib'), independent_final_test=False, default_changed=False,
        all_five_transfer_pending=True, durable_dense_prediction_bytes=0, study_bytes=tree_bytes(STUDY))
    run.json('summary.json', summary)
    print(json.dumps({k: {m: r['scene_macro'] for m, r in v['by_modality'].items()} for k, v in result.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
