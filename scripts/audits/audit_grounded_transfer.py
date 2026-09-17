"""Cold state, real query gate and fixed-index qualitative transfer audit."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.v4.evaluation.lerf_common import load_labels, ground_truth_masks
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--device', required=True)
    a = parser.parse_args()
    torch.set_num_threads(4)
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    root = base / 'preserved_source_inventory_20260916' / a.scene
    output = Path(f'paper/artifacts/{a.scene}_grounded_transfer_audit_20260916.json')
    if output.exists():
        raise FileExistsError(output)
    names = ['full_source_query_threshold_control', 'grounded_semantic_transfer96']
    gr = json.loads((base / 'geometry_only_repair_final_20260914' / a.scene / 'report.json').read_text())
    rec = gr['results']['independent_scale']
    geo = FrozenGaussianGeometry.load(rec['geometry'], expected_sha256=rec['geometry_sha256'], device=a.device)
    prior = MogeGeometryPrior(gr['source_report'], gr['source_report_sha256'])
    identity = json.loads((root / 'native_query_encoder.json').read_text())
    fields, payloads = [], []
    for name in names:
        report = json.loads((root / name / 'report.json').read_text())
        field, payload = load_element_retrieval_field(root / name / 'field.pt', expected_sha256=report['field_sha256'], geometry=geo, expected_query_encoder=identity, device=a.device)
        fields.append(field)
        payloads.append(payload)
    for key, value in payloads[0]['state_dict'].items():
        assert torch.equal(value, payloads[1]['state_dict'][key]), key
    ann, categories, sh, sw = load_labels(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label'), a.scene)
    encoder = NativeThreeQueryEncoder(identity, a.device)
    queries = encoder.encode_text(categories).half().float()
    old, new = fields
    runs = [json.loads(Path(f'paper/artifacts/{a.scene}_{kind}_text_20260916.json').read_text())['runs'][name]
            for kind, name in zip(['caption_transfer', 'grounded_transfer'], names)]
    references = [json.loads(Path(f'paper/artifacts/{a.scene}_{kind}_reference_20260916.json').read_text())['runs'][name]
                  for kind, name in zip(['caption_transfer', 'grounded_transfer'], names)]
    assert len(references[0]['observations']) == len(references[1]['observations'])
    for before, after in zip(references[0]['observations'], references[1]['observations']):
        for key in ['reference', 'target_frame', 'category', 'image', 'prompt']:
            assert before[key] == after[key], key
    with torch.no_grad():
        assert torch.equal(old.native_cosine(queries), new.native_cosine(queries))
        assert torch.equal(old.retrieve(queries, 'image')['scores'], new.retrieve(queries, 'image')['scores'])
        similarities = F.normalize(queries, dim=-1) @ F.normalize(new.grounded_transfer.prototypes, dim=-1).T
        confidence, indices = similarities.max(1)
        accepted = confidence >= new.grounded_transfer.acceptance[indices]
        scores = [f.retrieve(queries, 'text')['scores'] for f in fields]
        assert torch.equal(scores[0][:, ~accepted], scores[1][:, ~accepted])
        assert not scores[1][~new.observed].any()
        groups = json.loads((root / names[1] / 'report.json').read_text())['groups']
        gates = [dict(query=c, source_phrase=groups[int(indices[i])]['phrase'], cosine=float(confidence[i]),
                      acceptance=float(new.grounded_transfer.acceptance[indices[i]]), accepted=bool(accepted[i]),
                      before_iou=runs[0]['per_category_iou'][c], after_iou=runs[1]['per_category_iou'][c])
                 for i, c in enumerate(categories)]
        # Fixed alphabetic quarter indices, first annotated frame containing each.
        chosen = sorted({0, len(categories)//4, len(categories)//2, 3*len(categories)//4, len(categories)-1})
        pictures = []
        width = 480
        height = round(sh*width/sw)
        colors = torch.cat(scores + [old.observed.float()[:, None]], 1)
        for ci in chosen:
            category = categories[ci]
            frame = next(o['frame'] for o in runs[0]['observations'] if o['category'] == category)
            camera = prior.cameras[frame]
            k = camera.intrinsic.to(a.device).clone()
            k[0] *= width/camera.width
            k[1] *= height/camera.height
            image, alpha, _ = rasterization(means=geo.xyz, quats=geo.quaternions, scales=geo.scales,
                opacities=geo.opacities.reshape(-1), colors=colors, viewmats=camera.world_to_camera.to(a.device)[None],
                Ks=k[None], width=width, height=height, near_plane=.001*prior.scale, far_plane=100*prior.scale, packed=False)
            mass = image[0, ..., -1]
            valid = (mass/alpha[0, ..., 0].clamp_min(1e-8) > .5) & (alpha[0, ..., 0] > 1e-8)
            rendered = image[0, ..., [ci, len(categories)+ci]]/mass[..., None].clamp_min(1e-8)
            pred = ((rendered > .5) & valid[..., None]).cpu()
            gt = ground_truth_masks(ann[frame], [category], height, width, sh, sw)[category]
            measured = [float((pred[..., i] & gt).sum()/(pred[..., i] | gt).sum().clamp_min(1)) for i in range(2)]
            for i in range(2):
                expected = next(o['iou'] for o in runs[i]['observations'] if o['category'] == category and o['frame'] == frame)
                assert abs(expected-measured[i]) < 1e-6
            pictures.append((category, frame, gt, pred, measured))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(pictures), 3, figsize=(10, 2.6*len(pictures)), squeeze=False)
    for row, (category, frame, gt, pred, measured) in enumerate(pictures):
        for col, (img, label) in enumerate([(gt, 'GT'), (pred[..., 0], f'Before {measured[0]*100:.1f}%'), (pred[..., 1], f'After {measured[1]*100:.1f}%')]):
            axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row, col].set_title(f'{category} / frame {frame}: {label}', fontsize=10)
            axes[row, col].axis('off')
    fig.suptitle(a.scene + ': fixed alphabetical indices, including failures; IoU')
    fig.tight_layout()
    plot = output.with_suffix('.png')
    fig.savefig(plot, dpi=120)
    plt.close(fig)
    result = dict(schema='grounded_transfer_audit.v1', scene=a.scene, all_parent_tensors_exact=True,
        cold_native_cosine_exact=True, cold_image_response_exact=True, reference_image_prompt_observations_exact=True,
        fallback_query_response_exact=True, unsupported_scores_zero=True, gates=gates,
        selection='alphabetical quarter indices; first annotated frame containing category',
        plot=str(plot), plot_sha256=sha256_file(plot), implementation_sha256=sha256_file(__file__))
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(a.scene, sum(g['accepted'] for g in gates), '/', len(gates), flush=True)


if __name__ == '__main__':
    main()
