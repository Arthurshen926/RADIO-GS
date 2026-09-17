"""Fixed-geometry causal control of nested source-region semantic mixing.

Build happens before opening benchmark annotations. Controls are native-only,
not deployable replacements for a field whose structural heads used old features.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.training.region_visibility import normalize_visible_region_mass
from radio_gs.training.source_region_hierarchy import source_region_hierarchy, exclusive_scope_masks
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.v4.evaluation.lerf_common import load_labels, ground_truth_masks, binary_iou
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--scene', required=True, choices=['figurines', 'ramen'])
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError(args.output)
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    root = base / 'preserved_source_inventory_20260916' / args.scene
    state_path = root / 'source_scope_native_controls_20260916.pt'
    if state_path.exists():
        raise FileExistsError(state_path)
    info = json.loads((root / 'teachers.json').read_text())
    authority = json.loads((root / 'responsibility.json').read_text())
    episode = root / 'episodes.pt'
    assert sha256_file(episode) == json.loads(episode.with_suffix('.json').read_text())['sha256']
    data = torch.load(episode, weights_only=False)
    assert data['teacher_manifest_sha256'] == sha256_file(root / 'teachers.json')
    record = info['teacher_files']['siglip']
    assert sha256_file(root / record['path']) == record['sha256']
    teacher = torch.load(root / record['path'], weights_only=False)
    fp = root / 'exact_render_query_threshold96/field.pt'
    assert sha256_file(fp) == json.loads((fp.parent / 'report.json').read_text())['field_sha256']
    payload = torch.load(fp, weights_only=False)
    field = ElementRetrievalField(**payload['architecture']).to(args.device)
    field.load_state_dict(payload['state_dict'])
    field.requires_grad_(False)
    report = json.loads((base / 'geometry_only_repair_final_20260914' / args.scene / 'report.json').read_text())
    prior = MogeGeometryPrior(report['source_report'], report['source_report_sha256'])
    geometry_record = report['results']['independent_scale']
    geometry = FrozenGaussianGeometry.load(geometry_record['geometry'], expected_sha256=geometry_record['geometry_sha256'], device=args.device)
    for binding in [data['geometry_binding'], payload['geometry_binding'], authority['metadata']['frozen_geometry_binding']]:
        geometry.assert_binding(binding)
    views = data['proposal_views'].to(args.device)
    train = torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in views], device=args.device)
    # Heldout-source masks cannot establish any hierarchy or overwrite source rows.
    masks = data['masks'].to(args.device).clone()
    masks[~train] = False
    relation = source_region_hierarchy(masks, views)
    controls = {'parent_native': masks, **{scope: exclusive_scope_masks(masks, relation, scope=scope) for scope in ['fine', 'coarse']}}
    descriptors = F.normalize(teacher['descriptors'].float().to(args.device), dim=1)
    projected = descriptors @ field.native_decoder.weight
    n = len(geometry.xyz)
    accum = {name: torch.zeros(n, 64, device=args.device) for name in controls}
    masses = {name: torch.zeros(n, device=args.device) for name in controls}
    transport_error = 0.
    assert authority['metadata']['feature_height'] == masks.shape[1]
    assert authority['metadata']['feature_width'] == masks.shape[2]
    for record_view in authority['views']:
        frame = record_view['frame_index']
        if frame not in info['train_frames']:
            continue
        vi = info['frames'].index(frame)
        selected = torch.where(views == vi)[0]
        path = root / record_view['relative_path']
        assert sha256_file(path) == record_view['sha256']
        hit = torch.load(path, map_location=args.device, weights_only=False)
        pid, gid, w = hit['pixel_ids'].long(), hit['gaussian_ids'].long(), hit['base_weights'].float()
        pixels = masks.shape[1] * masks.shape[2]
        pixel_mass = torch.zeros(pixels, device=args.device).scatter_add_(0, pid, w)
        mat = torch.sparse_coo_tensor(torch.stack([pid, gid]), w / pixel_mass[pid].clamp_min(1e-8), (pixels, n)).coalesce()
        row_mass = torch.sparse.sum(mat, dim=0).to_dense()
        for name, target in controls.items():
            lifted, _ = normalize_visible_region_mass(torch.sparse.mm(mat.T, target[selected].flatten(1).float().T), row_mass)
            weights = lifted.T.half().float()
            if name == 'parent_native':
                transport_error = max(transport_error, float((weights - data['region_probability'][selected.cpu()].to(args.device)).abs().max()))
                # Sparse accumulation order can straddle an FP16 rounding bin.
                # Keep the baseline bit-identical to its sealed training input.
                weights = data['region_probability'][selected.cpu()].float().to(args.device)
            accum[name] += weights.T @ projected[selected]
            masses[name] += weights.sum(0)
    assert transport_error <= 2**-11, transport_error
    latents = {name: values / masses[name][:, None].clamp_min(1e-8) for name, values in accum.items()}
    torch.testing.assert_close(latents['parent_native'][field.observed], field.latent[field.observed, :64], atol=2e-5, rtol=2e-4)
    for name in controls:
        assert (masses[name][field.observed] > 0).all(), 'control lost observed support'
    pairs = relation.nonzero()
    pair_cosine = (descriptors[pairs[:, 0]] * descriptors[pairs[:, 1]]).sum(1)
    stats = dict(source_regions=int(train.sum()), nested_relations=len(pairs), regions_in_relations=int((relation.any(0) | relation.any(1)).sum()),
                 nested_descriptor_cosine_median=float(pair_cosine.median()) if len(pairs) else None,
                 nested_descriptor_cosine_p10=float(pair_cosine.quantile(.1)) if len(pairs) else None,
                 transport_max_error=transport_error, pixels_removed={name: int(masks.sum()-target.sum()) for name, target in controls.items()},
                 affected_element_fraction={name: float(((latents[name]-latents['parent_native']).norm(dim=1)[field.observed]>1e-5).float().mean()) for name in controls})
    source_artifact = dict(schema='source_scope_native_controls.v1', diagnostic_only=True, benchmark_data_used=False,
                           geometry_binding=geometry.fingerprint(), source_episode_sha256=sha256_file(episode),
                           teacher_sha256=info['teacher_files']['siglip']['sha256'], authority_sha256=sha256_file(root/'responsibility.json'),
                           basis=field.native_decoder.weight.cpu(), valid=field.observed.cpu(), latents={k:v.cpu() for k,v in latents.items()},
                           hierarchy=relation.cpu(), stats=stats, train_frames=info['train_frames'])
    torch.save(source_artifact, state_path)
    print(args.scene, 'source controls sealed', stats, flush=True)
    # Only metrics below access benchmark queries/masks; no source build or fit follows.
    annotations, categories, sh, sw = load_labels(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label'), args.scene)
    assert not set(annotations) & set(info['train_frames'])
    encoder = NativeThreeQueryEncoder(payload['query_encoder'], args.device)
    q = encoder.encode_text(categories).half().float()
    threshold = field.native_query_thresholds['text'](q)
    scale = field.readout_log_scale['text'].clamp(-4, 6).exp()
    # Orthonormal zero-bias decoder: norm(Dz)=norm(z), avoiding dense N*1536 allocation.
    basis = field.native_decoder.weight
    torch.testing.assert_close(basis.T @ basis, torch.eye(64, device=args.device), atol=1e-4, rtol=1e-4)
    assert not field.native_decoder.bias.any()
    query = F.normalize(q, dim=-1) @ basis
    responses = {name: torch.sigmoid(scale*(F.normalize(latent, dim=-1) @ query.T-threshold)) for name, latent in latents.items()}
    torch.testing.assert_close(responses['parent_native'][field.observed], torch.sigmoid(scale*(field.native_cosine(q)-threshold))[field.observed], atol=2e-5, rtol=2e-4)
    rows = {name: [] for name in responses}
    height = round(sh*480/sw)
    for frame, objects in sorted(annotations.items()):
        cam = prior.cameras[frame]
        k = cam.intrinsic.to(args.device).clone()
        k[0] *= 480/cam.width
        k[1] *= height/cam.height
        targets = ground_truth_masks(objects, categories, height, 480, sh, sw)
        for name, response in responses.items():
            colors = torch.cat([response*field.observed[:, None], field.observed.float()[:, None]], 1)
            image, alpha, _ = rasterization(means=geometry.xyz, quats=geometry.quaternions, scales=geometry.scales, opacities=geometry.opacities.flatten(), colors=colors, viewmats=cam.world_to_camera.to(args.device)[None], Ks=k[None], width=480, height=height, near_plane=.001*prior.scale, far_plane=100*prior.scale, packed=False)
            support = image[0, ..., -1]
            known = (support/alpha[0, ..., 0].clamp_min(1e-8)>.5) & (alpha[0, ..., 0]>1e-8)
            score = image[0, ..., :len(q)]/support[..., None].clamp_min(1e-8)
            for ci, category in enumerate(categories):
                truth = targets[category].to(args.device)
                if not truth.any():
                    continue
                value = torch.where(known, score[..., ci], torch.zeros_like(support))
                iou, _, _ = binary_iou((value>.5)&known, truth)
                rows[name].append(dict(frame=frame, category=category, iou=iou, **binary_ranking_metrics(value, truth)))
    macro = {name: {key: sum(row[key] for row in rr)/len(rr) for key in ['iou', 'average_precision']} for name, rr in rows.items()}
    result = dict(schema='source_scope_fusion_diagnostic.v1', scene=args.scene, diagnostic_only=True, complete_method=False,
                  geometry_unchanged=True, source_descriptors_and_basis_unchanged=True, source_only_assignment=True,
                  inherited_threshold=True, benchmark_masks_for_metrics_only=True, target_rgb_used=False,
                  parameters=dict(containment=.95, max_area_ratio=.8), stats=stats, macro=macro, rows=rows,
                  parent_field_sha256=sha256_file(fp), source_controls_path=str(state_path), source_controls_sha256=sha256_file(state_path),
                  implementation_sha256=sha256_file(__file__), hierarchy_implementation_sha256=sha256_file(source_region_hierarchy.__code__.co_filename))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(args.scene, macro, flush=True)


if __name__ == '__main__':
    with torch.no_grad():
        main()
