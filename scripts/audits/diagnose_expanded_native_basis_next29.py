"""Post-hoc, single-scene failure localization; never a benchmark candidate."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.native_decoder_readout import affine_decoder_cosine
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels, volume_weighted_split_metrics
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4)
    scene = 'scene0140_00'
    root = Path('output/scannet_expanded_budget_next29')/scene
    field = root/'reconstruction/field'
    old = Path('output/scannet_coverage_native_next28')/scene
    out = root/'native_basis_diagnostic'
    out.mkdir(exist_ok=False)
    saved = False

    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            if '.labels.ply' in str(args[0]) and not saved:
                raise RuntimeError('GT opened before all diagnostic predictions')

    sys.addaudithook(guard)
    info = json.loads((field/'teachers.json').read_text())

    def load(path, digest):
        assert sha256_file(path) == digest
        return torch.load(path, map_location='cpu', weights_only=False)

    ep_receipt = json.loads((field/'episodes.json').read_text())
    ep = load(field/'episodes.pt', ep_receipt['sha256'])
    assert ep['teacher_manifest_sha256'] == sha256_file(field/'teachers.json')
    authority = json.loads((field/'responsibility.json').read_text())
    assert info['authority_sha256'] == sha256_file(field/'responsibility.json')
    tr = info['teacher_files']['siglip']
    descriptors = F.normalize(load(field/tr['path'], tr['sha256'])['descriptors'].float().cuda(), dim=-1)
    current_receipt = json.loads((field/'bounded_fallback_admission/report.json').read_text())
    current = load(field/'bounded_fallback_admission/field.pt', current_receipt['field_sha256'])['state_dict']
    old_field = old/'reconstruction_complete_next28/field/bounded_fallback_admission'
    old_receipt = json.loads((old_field/'report.json').read_text())
    previous = load(old_field/'field.pt', old_receipt['field_sha256'])['state_dict']
    assert json.loads((old_field.parent/'native_query_encoder.json').read_text()) == json.loads((field/'native_query_encoder.json').read_text())
    n = len(current['latent'])
    numerator = torch.zeros((n, descriptors.shape[1]), device='cuda')
    mass = torch.zeros(n, device='cuda')
    for frame in info['train_frames']:
        vi = info['frames'].index(frame)
        ids = torch.where(ep['proposal_views'] == vi)[0]
        rec = authority['views'][vi]
        hit = load(field/rec['relative_path'], rec['sha256'])
        weights, _ = source_pixel_evidence_weights(ep['masks'][ids].flatten(1).cuda(),
            hit['pixel_ids'].long().cuda(), hit['gaussian_ids'].long().cuda(),
            hit['base_weights'].float().cuda(), n, partition_overlap=True)
        numerator.addmm_(weights.T, descriptors[ids])
        mass += weights.sum(0)
        print('source frame', frame, flush=True)
    features = numerator / mass[:, None].clamp_min(1e-8)
    observed = current['observed'].cuda()
    assert (mass[observed] > 0).all() and torch.isfinite(features).all()
    # The feature construction above is query independent. Only now read the
    # saved benchmark encodings for this diagnostic inference.
    reference = json.loads((root/'evaluation/report.json').read_text())
    q = load(root/'evaluation/queries.pt', reference['query_sha256'])['encoding'].cuda()
    zeros = torch.zeros(features.shape[1], device='cuda')
    scores = {'uncompressed1536': F.normalize(features, dim=-1) @ F.normalize(q, dim=-1).T}
    for name, state in [('new_basis64', current), ('old_basis64', previous)]:
        weight = state['native_decoder.weight'].cuda()
        assert torch.count_nonzero(state['native_decoder.bias']) == 0
        scores[name] = affine_decoder_cosine(features @ weight, q, weight, zeros)
    original_scores = affine_decoder_cosine(current['latent'][:, :64].cuda(), q,
                                            current['native_decoder.weight'].cuda(), zeros)
    error = float((scores['new_basis64'][observed]-original_scores[observed]).abs().max())
    assert error < 2e-5, error
    classes = sorted(set(sum([SPLITS[s] for s in ('19', '15', '10')], [])))
    predictions = {}
    for name, score in scores.items():
        predictions[name] = {}
        for split in ('19', '15', '10'):
            labels = SPLITS[split]
            columns = [classes.index(c) for c in labels]
            pred = torch.tensor(labels, device='cuda')[score[:, columns].argmax(1)]
            predictions[name][split] = torch.where(observed, pred, torch.zeros_like(pred)).cpu().numpy()
        np.savez_compressed(out/(name+'.npz'), **predictions[name])
    original = np.load(root/'evaluation/native.npz')
    mismatches = {s: int(np.count_nonzero(original[s] != predictions['new_basis64'][s])) for s in ('19', '15', '10')}
    saved = True
    gr = json.loads((root/'geometry_report.json').read_text())['results']['independent_scale']
    assert gr['geometry_sha256'] == reference['geometry_sha256']
    geo = FrozenGaussianGeometry.load(gr['geometry'], expected_sha256=gr['geometry_sha256'], device='cpu')
    gt = Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/(scene+'_vh_clean_2.labels.ply')
    assert sha256_file(gt) == reference['gt_sha256']
    xyz, labels = _read_label_ply(gt)
    pseudo, audit = assign_vala_pseudo_labels(geo.xyz.numpy(), geo.scales.numpy(), geo.quaternions.numpy(),
        xyz, labels, radius_factor=5., candidate_k=1000, fallback_k=1, class_balance=True, chunk_size=512)
    assert audit == reference['pseudo_gt_audit']
    weights = (geo.scales.prod(1)*geo.opacities.flatten()).numpy()
    metrics = {name: {s: volume_weighted_split_metrics(pseudo, pred[s], weights, SPLITS[s]) for s in ('19', '15', '10')}
               for name, pred in predictions.items()}
    report = dict(scene=scene, diagnostic_only=True, post_hoc_scene_selection=True,
        scene_selection_reason='Native10-class regression after expanded reconstruction; not a candidate or eight-scene replacement.',
        geometry_sha256=gr['geometry_sha256'], original_evaluation_sha256=sha256_file(root/'evaluation/report.json'),
        teacher_manifest_sha256=sha256_file(field/'teachers.json'), source_frames=info['train_frames'],
        old_field_sha256=old_receipt['field_sha256'], current_field_sha256=current_receipt['field_sha256'],
        query_sha256=reference['query_sha256'], prediction_sha256={k: sha256_file(out/(k+'.npz')) for k in predictions},
        gt_sha256=reference['gt_sha256'], gt_opened_after_predictions=True,
        new_basis_score_roundoff_max=error, new_basis_prediction_mismatches=mismatches, metrics=metrics,
        compatible_with_current_heads=False, implementation_sha256=sha256_file(__file__))
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print({k: {s: round(v[s]['miou']*100, 4) for s in ('19', '15', '10')} for k, v in metrics.items()}, flush=True)


if __name__ == '__main__':
    main()
