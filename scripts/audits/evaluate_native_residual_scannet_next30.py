"""Cold category control: original native versus source-admitted native residual."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scannet_constants import NYU40_ID_TO_NAME, OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels, volume_weighted_split_metrics
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    saved = False
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if '.labels.ply' in path and not saved:
                raise RuntimeError('GT before predictions')
            if any(x in path for x in ('teacher_siglip.pt', 'teacher_sam.pt', 'teacher_dino.pt', 'teacher_radio.pt', 'episodes.pt')):
                raise RuntimeError('Training assets forbidden at query time')
    sys.addaudithook(guard)
    for scene in a.scenes:
        saved = False
        folder = Path('output/native_residual_basis_next30_v2')/('scannet_'+scene)
        map_report = json.loads((folder/'map/report.json').read_text())
        out = folder/'category_evaluation'
        out.mkdir(exist_ok=False)
        fieldroot = Path(map_report['root'])
        root = fieldroot.parent.parent
        expanded = 'scannet_expanded_budget_next29' in str(root)
        reference_dir = root/('evaluation' if expanded else 'evaluation_complete_next28')
        reference = json.loads((reference_dir/'report.json').read_text())
        assert sha256_file(folder/'map/map.pt') == map_report['map_sha256']
        data = torch.load(folder/'map/map.pt', map_location='cuda', weights_only=False)
        gr = json.loads((root/'geometry_report.json').read_text())['results']['independent_scale']
        geo = FrozenGaussianGeometry.load(gr['geometry'], expected_sha256=gr['geometry_sha256'], device='cpu')
        geo.assert_binding(data['geometry_binding'])
        qp = reference_dir/'queries.pt'
        assert sha256_file(qp) == reference['query_sha256']
        queries = torch.load(qp, map_location='cpu', weights_only=False)
        assert queries['encoder_identity'] == json.loads((fieldroot/'native_query_encoder.json').read_text())
        classes = sorted(set(sum([SPLITS[s] for s in ('19', '15', '10')], [])))
        assert queries['texts'] == [NYU40_ID_TO_NAME[c] for c in classes]
        q = F.normalize(queries['encoding'].cuda(), dim=-1)
        features, basis = data['features'], data['basis']
        scores = {name: F.normalize(features[:, :d], dim=-1) @ (q @ basis[:d].T).T
                  for name, d in [('native64', 64), ('native_residual', len(basis))]}
        predictions = {}
        for name, score in scores.items():
            predictions[name] = {}
            for split in ('19', '15', '10'):
                ids = SPLITS[split]
                selected = torch.tensor(ids, device='cuda')[score[:, [classes.index(c) for c in ids]].argmax(1)]
                predictions[name][split] = torch.where(data['observed'], selected, 0).cpu().numpy()
            np.savez_compressed(out/(name+'.npz'), **predictions[name])
        original = np.load(reference_dir/'native.npz')
        mismatches = {s: int(np.count_nonzero(original[s] != predictions['native64'][s])) for s in SPLITS}
        assert not any(mismatches.values()), mismatches
        saved = True
        gt = Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/(scene+'_vh_clean_2.labels.ply')
        assert sha256_file(gt) == reference['gt_sha256']
        xyz, labels = _read_label_ply(gt)
        pseudo, audit = assign_vala_pseudo_labels(geo.xyz.numpy(), geo.scales.numpy(), geo.quaternions.numpy(), xyz, labels,
            radius_factor=5., candidate_k=1000, fallback_k=1, class_balance=True, chunk_size=512)
        weights = (geo.scales.prod(1)*geo.opacities.flatten()).numpy()
        metrics = {name: {s: volume_weighted_split_metrics(pseudo, pred[s], weights, SPLITS[s]) for s in ('19','15','10')}
                   for name, pred in predictions.items()}
        report = dict(scene=scene, diagnostic_native_readout=True, mainline_promoted=False, independent_test=False,
            final_scope_heads_reused=False, native_baseline_prediction_mismatches=mismatches, metrics=metrics,
            map_report_sha256=sha256_file(folder/'map/report.json'), source_selection_sha256=sha256_file(folder/'report.json'),
            geometry_sha256=gr['geometry_sha256'], query_sha256=sha256_file(qp), gt_sha256=sha256_file(gt),
            gt_opened_after_predictions=True, pseudo_gt_audit=audit,
            predictions={k: sha256_file(out/(k+'.npz')) for k in predictions},
            implementation_sha256=sha256_file(__file__), metric_sha256=sha256_file('radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py'))
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(scene, {k: {s: round(v[s]['miou']*100, 3) for s in ('19','15','10')} for k,v in metrics.items()}, flush=True)


if __name__ == '__main__':
    main()
