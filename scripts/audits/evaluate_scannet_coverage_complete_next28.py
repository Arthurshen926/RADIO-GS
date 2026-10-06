"""Evaluate one complete rebuilt ScanNet field with the frozen category protocol."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.querying.bounded_native_scope import refine_native_scope_logits
from radio_gs.scannet_constants import NYU40_ID_TO_NAME, OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels, volume_weighted_split_metrics
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--attempt-name', default='complete_next28')
    args = parser.parse_args()
    torch.set_num_threads(4)
    scene = args.scene
    root = (Path('output/scannet_coverage_native_probe_20260919') if scene in ('scene0000_00', 'scene0062_00')
            else Path('output/scannet_coverage_native_next28')) / scene
    if not args.attempt_name.replace('_', '').isalnum():
        raise ValueError('attempt-name must contain only letters, digits or underscore')
    execution_path = root/(args.attempt_name+'_execution.json')
    execution = json.loads(execution_path.read_text())
    if not execution['complete']:
        raise RuntimeError('Complete field reconstruction not finished')
    output = root/'evaluation_complete_next28'
    output.mkdir(exist_ok=False)
    gtpath = Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/(scene+'_vh_clean_2.labels.ply')
    predictions_saved = False

    def guard(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            path = str(arguments[0])
            if '.labels.ply' in path and not predictions_saved:
                raise RuntimeError('Semantic ground truth opened before predictions')
            if any(name in path for name in ('teacher_siglip.pt', 'teacher_radio.pt', 'teacher_dino.pt',
                                             'teacher_sam.pt', 'episodes.pt', 'caption_region_evidence.pt')):
                raise RuntimeError('Training asset read during query evaluation')

    sys.addaudithook(guard)
    prior = json.loads((root/'geometry_report.json').read_text())
    geometry_record = prior['results']['independent_scale']
    geometry = FrozenGaussianGeometry.load(geometry_record['geometry'],
                                           expected_sha256=geometry_record['geometry_sha256'], device=args.device)
    field_root = root/'reconstruction_complete_next28/field'
    run = 'bounded_fallback_admission'
    model_record = json.loads((field_root/run/'report.json').read_text())
    identity = json.loads((field_root/'native_query_encoder.json').read_text())
    model, payload = load_element_retrieval_field(field_root/run/'field.pt',
                                                   expected_sha256=model_record['field_sha256'],
                                                   geometry=geometry, expected_query_encoder=identity,
                                                   device=args.device)
    geometry.assert_binding(payload['geometry_binding'])
    classes = sorted(set(sum([SPLITS[split] for split in ('19', '15', '10')], [])))
    texts = [NYU40_ID_TO_NAME[class_id] for class_id in classes]
    encoder = NativeThreeQueryEncoder(identity, args.device)
    with torch.no_grad():
        queries = encoder.encode_text(texts).half().float()
        cosine = model.native_cosine(queries)
        scale = model.readout_log_scale['text'].clamp(-4, 6).exp()
        calibrated = scale*(cosine-model.native_query_thresholds['text'](queries))
        scope = model.logits(queries, 'text')
        scores = {'native': cosine,
                  'bounded_scope': refine_native_scope_logits(calibrated, scope, max_correction=1.0)}
        predictions = {}
        for name, score in scores.items():
            predictions[name] = {}
            for split in ('19', '15', '10'):
                labels = SPLITS[split]
                columns = [classes.index(class_id) for class_id in labels]
                selected = torch.tensor(labels, device=args.device)[score[:, columns].argmax(1)]
                predictions[name][split] = torch.where(model.observed, selected,
                                                        torch.zeros_like(selected)).cpu().numpy()
            np.savez_compressed(output/(name+'.npz'), **predictions[name])
        torch.save(dict(texts=texts, encoding=queries.cpu(), encoder_identity=identity), output/'queries.pt')
    predictions_saved = True
    xyz, labels = _read_label_ply(gtpath)
    pseudo, audit = assign_vala_pseudo_labels(geometry.xyz.cpu().numpy(),
                                               geometry.scales.cpu().numpy(), geometry.quaternions.cpu().numpy(),
                                               xyz, labels, radius_factor=5., candidate_k=1000,
                                               fallback_k=1, class_balance=True, chunk_size=512)
    weights = (geometry.scales.prod(1)*geometry.opacities.flatten()).cpu().numpy()
    metrics = {name: {split: volume_weighted_split_metrics(pseudo, by_split[split], weights, SPLITS[split])
                      for split in ('19', '15', '10')}
               for name, by_split in predictions.items()}
    baseline = json.loads((Path('output/scannet_frozen_field_20260918')/scene/
                           'official_protocol_evaluation/report.json').read_text())
    report = dict(scene=scene, contract='scannet-ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test=False,
                  complete_field_rebuilt=True, source_view_budget=16,
                  geometry_sha256=geometry_record['geometry_sha256'],
                  field_sha256=model_record['field_sha256'],
                  execution_sha256=sha256_file(execution_path),
                  source_acceptance=execution.get('source_acceptance'),
                  gt_sha256=sha256_file(gtpath), gt_opened_after_prediction=True,
                  query_sha256=sha256_file(output/'queries.pt'),
                  prediction_sha256={name: sha256_file(output/(name+'.npz')) for name in predictions},
                  baseline_metrics=baseline['metrics'], metrics=metrics,
                  bounded_scope_rule='calibrated_native_logit + clamp(scope_logit - calibrated_native_logit, -1, 1)',
                  rule_fixed_before_target_gt=True, pseudo_gt_audit=audit,
                  metric_implementation_sha256=sha256_file('radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py'),
                  implementation_sha256=sha256_file(__file__))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(scene, {name: {split: round(metrics[name][split]['miou']*100, 3)
                          for split in ('19', '15', '10')} for name in metrics}, flush=True)


if __name__ == '__main__':
    main()
