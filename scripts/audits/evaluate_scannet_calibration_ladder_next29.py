"""Post-hoc decomposition of the fixed native-to-bounded ScanNet score rule."""
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
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels, volume_weighted_split_metrics
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    root = Path('output/scannet_expanded_budget_next29') / a.scene
    reference_path = root / 'evaluation/report.json'
    reference = json.loads(reference_path.read_text())
    assert reference['scene'] == a.scene and reference['complete_field_rebuilt']
    out = root / 'calibration_ladder'
    out.mkdir(exist_ok=False)
    saved = False

    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if '.labels.ply' in path and not saved:
                raise RuntimeError('Target labels opened before predictions')
            if any(name in path for name in ('teacher_siglip.pt', 'teacher_radio.pt',
                                             'teacher_dino.pt', 'teacher_sam.pt',
                                             'episodes.pt', 'caption_region_evidence.pt')):
                raise RuntimeError('Training asset read during score decomposition')
    sys.addaudithook(guard)
    prior = json.loads((root / 'geometry_report.json').read_text())
    gr = prior['results']['independent_scale']
    geo = FrozenGaussianGeometry.load(gr['geometry'], expected_sha256=gr['geometry_sha256'],
                                      device=a.device)
    field_root = root / 'reconstruction/field'
    name = 'bounded_fallback_admission'
    field_report = json.loads((field_root / name / 'report.json').read_text())
    identity = json.loads((field_root / 'native_query_encoder.json').read_text())
    field, payload = load_element_retrieval_field(
        field_root / name / 'field.pt', expected_sha256=field_report['field_sha256'],
        geometry=geo, expected_query_encoder=identity, device=a.device)
    geo.assert_binding(payload['geometry_binding'])
    assert field_report['field_sha256'] == reference['field_sha256']
    query_path = root / 'evaluation/queries.pt'
    assert sha256_file(query_path) == reference['query_sha256']
    saved_queries = torch.load(query_path, map_location='cpu', weights_only=False)
    assert saved_queries['encoder_identity'] == identity
    queries = saved_queries['encoding'].to(a.device)
    classes = sorted(set(sum([SPLITS[s] for s in ('19', '15', '10')], [])))
    assert len(classes) == len(saved_queries['texts'])
    with torch.no_grad():
        cosine = field.native_cosine(queries)
        scale = field.readout_log_scale['text'].clamp(-4, 6).exp()
        calibrated = scale * (cosine - field.native_query_thresholds['text'](queries))
        scope_receipt = json.loads((field_root/'bounded_fallback_withheld/report.json').read_text())
        assert sha256_file(field_root/'bounded_fallback_withheld/field.pt') == scope_receipt['field_sha256']
        bounded, decision = source_accepted_category_logits(calibrated,
            lambda: field.logits(queries, 'text'), scope_receipt=scope_receipt,
            admission_receipt=field_report, max_correction=1.0)
        assert decision == reference['deployment_decision']
        predictions = {}
        for readout, scores in {'native': cosine, 'calibrated_native': calibrated,
                                'source_accepted': bounded}.items():
            predictions[readout] = {}
            for split in ('19', '15', '10'):
                ids = SPLITS[split]
                columns = [classes.index(c) for c in ids]
                selected = torch.tensor(ids, device=a.device)[scores[:, columns].argmax(1)]
                predictions[readout][split] = torch.where(field.observed, selected,
                                                          torch.zeros_like(selected)).cpu().numpy()
            np.savez_compressed(out / (readout + '.npz'), **predictions[readout])
    for readout in ('native', 'source_accepted'):
        original = np.load(root / 'evaluation' / (readout + '.npz'))
        assert all(np.array_equal(predictions[readout][s], original[s])
                   for s in ('19', '15', '10'))
    saved = True
    gtpath = Path('/mnt/pool/sqy/3d_understanding/scannet_og') / a.scene / (
        a.scene + '_vh_clean_2.labels.ply')
    xyz, labels = _read_label_ply(gtpath)
    pseudo, audit = assign_vala_pseudo_labels(
        geo.xyz.cpu().numpy(), geo.scales.cpu().numpy(), geo.quaternions.cpu().numpy(),
        xyz, labels, radius_factor=5., candidate_k=1000, fallback_k=1,
        class_balance=True, chunk_size=512)
    assert audit == reference['pseudo_gt_audit']
    weights = (geo.scales.prod(1) * geo.opacities.flatten()).cpu().numpy()
    metrics = {readout: {
        split: volume_weighted_split_metrics(pseudo, by_split[split], weights, SPLITS[split])
        for split in ('19', '15', '10')}
        for readout, by_split in predictions.items()}
    for readout in ('native', 'source_accepted'):
        for split in ('19', '15', '10'):
            assert abs(metrics[readout][split]['miou'] -
                       reference['metrics'][readout][split]['miou']) < 1e-10
    report = dict(scene=a.scene, decomposition_only=True, post_hoc_diagnostic=True,
                  benchmark_arm=False, source_only_field=True,
                  target_gt_opened_after_predictions=True,
                  original_evaluation_sha256=sha256_file(reference_path),
                  field_sha256=field_report['field_sha256'],
                  query_sha256=sha256_file(query_path), gt_sha256=sha256_file(gtpath),
                  prediction_sha256={k: sha256_file(out / (k + '.npz')) for k in predictions},
                  metrics=metrics, pseudo_gt_audit=audit,
                  implementation_sha256=sha256_file(__file__))
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(a.scene, {k: {s: round(v[s]['miou'] * 100, 3) for s in ('19', '15', '10')}
                    for k, v in metrics.items()}, flush=True)


if __name__ == '__main__':
    main()
