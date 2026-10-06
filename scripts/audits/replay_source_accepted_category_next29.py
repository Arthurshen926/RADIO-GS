"""Execute the source gate before scope inference; score only persisted predictions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.querying.source_accepted_category import source_accepted_category_logits
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    torch.set_num_threads(4)
    root = (Path('output/scannet_coverage_native_probe_20260919') if a.scene in
            ('scene0000_00', 'scene0062_00') else Path('output/scannet_coverage_native_next28')) / a.scene
    out = Path('output/source_accepted_runtime_next29') / a.scene
    out.mkdir(parents=True, exist_ok=False)
    saved = False

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = str(args[0])
        if '.labels.ply' in path or any(x in path for x in
                ('teacher_siglip.pt', 'teacher_radio.pt', 'teacher_dino.pt',
                 'teacher_sam.pt', 'episodes.pt', 'caption_region_evidence.pt')):
            raise RuntimeError('Ground truth and training arrays forbidden in runtime replay')
        if not saved and path.endswith('.json') and any(x in path for x in
                ('evaluation_complete_next28', 'calibration_ladder_next28',
                 'complete_summary', 'source_gated_summary')):
            raise RuntimeError('Prior target metric reports forbidden before prediction')
    sys.addaudithook(guard)
    field_root = root / 'reconstruction_complete_next28/field'
    scope_path = field_root / 'bounded_fallback_withheld/report.json'
    admission_path = field_root / 'bounded_fallback_admission/report.json'
    scope_receipt = json.loads(scope_path.read_text())
    admission_receipt = json.loads(admission_path.read_text())
    assert sha256_file(scope_path.with_name('field.pt')) == scope_receipt['field_sha256']
    gr = json.loads((root / 'geometry_report.json').read_text())['results']['independent_scale']
    geometry = FrozenGaussianGeometry.load(gr['geometry'], expected_sha256=gr['geometry_sha256'], device=a.device)
    identity = json.loads((field_root / 'native_query_encoder.json').read_text())
    model, _ = load_element_retrieval_field(admission_path.with_name('field.pt'),
        expected_sha256=admission_receipt['field_sha256'], geometry=geometry,
        expected_query_encoder=identity, device=a.device)
    query_path = root / 'evaluation_complete_next28/queries.pt'
    bank = torch.load(query_path, map_location='cpu', weights_only=False)
    assert bank['encoder_identity'] == identity
    queries = bank['encoding'].to(a.device)
    calls = []

    def scope_forward():
        calls.append(True)
        return model.logits(queries, 'text')

    with torch.no_grad():
        native = model.readout_log_scale['text'].clamp(-4, 6).exp() * (
            model.native_cosine(queries) - model.native_query_thresholds['text'](queries))
        scores, decision = source_accepted_category_logits(native, scope_forward,
            scope_receipt=scope_receipt, admission_receipt=admission_receipt)
        classes = sorted(set(sum([SPLITS[s] for s in ('19', '15', '10')], [])))
        predictions = {}
        for split in ('19', '15', '10'):
            labels = SPLITS[split]
            cols = [classes.index(c) for c in labels]
            pred = torch.tensor(labels, device=a.device)[scores[:, cols].argmax(1)]
            predictions[split] = torch.where(model.observed, pred, torch.zeros_like(pred)).cpu().numpy()
    assert len(calls) == int(decision['use_scope'])
    prediction_path = out / 'predictions.npz'
    np.savez_compressed(prediction_path, **predictions)
    source_record = dict(scene=a.scene, decision=decision, scope_forward_calls=len(calls),
        scope_report_sha256=sha256_file(scope_path), admission_report_sha256=sha256_file(admission_path),
        field_sha256=admission_receipt['field_sha256'], geometry_sha256=gr['geometry_sha256'],
        query_sha256=sha256_file(query_path), prediction_sha256=sha256_file(prediction_path),
        ground_truth_read=False, target_reports_read_before_prediction=False,
        implementation_sha256=sha256_file(__file__),
        runtime_sha256=sha256_file('radio_gs/querying/source_accepted_category.py'))
    (out / 'source_decision.json').write_text(json.dumps(source_record, indent=2)+'\n')
    saved = True
    reference_root = root / ('evaluation_complete_next28' if decision['use_scope'] else 'calibration_ladder_next28')
    reference = json.loads((reference_root / 'report.json').read_text())
    name = decision['readout']
    reference_prediction = reference_root / (name + '.npz')
    assert sha256_file(reference_prediction) == reference['prediction_sha256'][name]
    expected = np.load(reference_prediction)
    mismatches = {s: int((predictions[s] != expected[s]).sum()) for s in predictions}
    assert not any(mismatches.values()), mismatches
    report = dict(source_record, complete=True, prediction_mismatches=mismatches,
        source_decision_sha256=sha256_file(out / 'source_decision.json'),
        reference_report_sha256=sha256_file(reference_root / 'report.json'),
        metrics=reference['metrics'][name], metrics_replayed_not_new_evaluation=True,
        independent_test=False, new_performance_gain=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(a.scene, decision, {s: round(report['metrics'][s]['miou']*100, 3) for s in predictions}, flush=True)


if __name__ == '__main__':
    main()
