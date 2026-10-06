"""Current field readout with the unchanged paper8 VALA Gaussian-domain metric."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.scannet_constants import NYU40_ID_TO_NAME, OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels, volume_weighted_split_metrics
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(4)
    root = Path('output/scannet_frozen_field_20260918')/a.scene
    assert json.loads((root/'execution.json').read_text())['complete']
    fieldroot = root/'reconstruction/field'
    run = 'bounded_fallback_admission'
    prior = json.loads((root/'geometry_report.json').read_text())
    g = prior['results']['independent_scale']
    gtpath = Path('/mnt/pool/sqy/3d_understanding/scannet_og')/a.scene/(a.scene+'_vh_clean_2.labels.ply')
    saved = False
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if '.labels.ply' in path and not saved:
                raise RuntimeError('Ground truth cannot precede persisted predictions')
            if any(s in path for s in ('teacher_siglip.pt', 'teacher_radio.pt', 'teacher_dino.pt', 'teacher_sam.pt',
                                      'episodes.pt', 'caption_region_evidence.pt', 'grounded_text_supervision.pt')):
                raise RuntimeError('Training evidence forbidden during evaluation')
    sys.addaudithook(guard)
    geo = FrozenGaussianGeometry.load(g['geometry'], expected_sha256=g['geometry_sha256'], device=a.device)
    identity = json.loads((fieldroot/'native_query_encoder.json').read_text())
    digest = json.loads((fieldroot/run/'report.json').read_text())['field_sha256']
    field, _ = load_element_retrieval_field(fieldroot/run/'field.pt', expected_sha256=digest,
        geometry=geo, expected_query_encoder=identity, device=a.device)
    classes = sorted(set(sum([OPENGAUSSIAN_NYU40_CLASS_SPLITS[s] for s in ['19', '15', '10']], [])))
    texts = [NYU40_ID_TO_NAME[c] for c in classes]
    encoder = NativeThreeQueryEncoder(identity, a.device)
    with torch.no_grad():
        query = encoder.encode_text(texts).half().float()
        # ScanNet predicts a semantic class across ALL its instances. The
        # existing object-specific scope route is a matched diagnostic, not
        # the category contract. Both are fixed before any GT is opened.
        response = field.native_cosine(query)
        object_response = field.retrieve(query, 'text')['scores']
        predictions = {}
        object_predictions = {}
        for split in ['19', '15', '10']:
            ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
            scores = response[:, [classes.index(c) for c in ids]]
            pred = torch.tensor(ids, device=a.device)[scores.argmax(1)]
            pred = torch.where(field.observed, pred, torch.zeros_like(pred))
            predictions[split] = pred.cpu().numpy()
            object_pred = torch.tensor(ids, device=a.device)[object_response[:, [classes.index(c) for c in ids]].argmax(1)]
            object_predictions[split] = torch.where(field.observed, object_pred, torch.zeros_like(object_pred)).cpu().numpy()
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'predictions.npz', **predictions)
    np.savez_compressed(a.output/'object_scope_control.npz', **object_predictions)
    torch.save(dict(texts=texts, encoding=query.cpu(), encoder_identity=identity), a.output/'queries.pt')
    saved = True
    xyz, labels = _read_label_ply(gtpath)
    pseudo, audit = assign_vala_pseudo_labels(geo.xyz.cpu().numpy(), geo.scales.cpu().numpy(),
        geo.quaternions.cpu().numpy(), xyz, labels, radius_factor=5., candidate_k=1000,
        fallback_k=1, class_balance=True, chunk_size=512)
    weights = geo.scales.prod(1).cpu().numpy()*geo.opacities.flatten().cpu().numpy()
    metrics = {s:volume_weighted_split_metrics(pseudo, predictions[s], weights,
        OPENGAUSSIAN_NYU40_CLASS_SPLITS[s]) for s in ['19', '15', '10']}
    object_metrics = {s:volume_weighted_split_metrics(pseudo, object_predictions[s], weights,
        OPENGAUSSIAN_NYU40_CLASS_SPLITS[s]) for s in ['19', '15', '10']}
    result = dict(contract='scannet-ovs-paper8-v1', scene=a.scene, input_track=prior['input_track'],
        independent_test=False, field_sha256=digest, geometry_sha256=g['geometry_sha256'],
        gt_sha256=sha256_file(gtpath), gt_opened_after_prediction=True,
        prediction_sha256=sha256_file(a.output/'predictions.npz'),
        query_sha256=sha256_file(a.output/'queries.pt'), pseudo_gt_audit=audit,
        readout='Semantic category: argmax frozen native teacher-space cosine over split classes; every instance eligible; unobserved elements predict void.',
        object_scope_control_sha256=sha256_file(a.output/'object_scope_control.npz'),
        object_scope_control_metrics=object_metrics,
        readout_selected_before_ground_truth=True,
        metrics=metrics, implementation_sha256=sha256_file(__file__),
        metric_implementation_sha256=sha256_file('radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py'))
    (a.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(a.scene, {s:(r['miou'], r['macc']) for s, r in metrics.items()}, flush=True)


if __name__ == '__main__':
    main()
