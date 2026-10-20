"""Actual selected-Gaussian subset fitting with explicit inputs only.

Numerical implementation extracted unchanged from the retained expanded-extent
solver. Legacy scene discovery and inherited prediction branches are removed.
All semantic-observed elements may update, including source-occluded elements.
"""
import argparse, json, time, shutil
from pathlib import Path
import torch
from radio_gs import evidence_core
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.querying.typed_evidence import visible_operator
from radio_gs.querying.finite_evidence import combine_with_native, source_strength
from radio_gs.querying.instance_reliability import instance_reliability

def _transport(operator, values):
    return evidence_core.project(operator, values) if evidence_core.enabled() else torch.sparse.mm(operator, values)


def load(p):
    return torch.load(p, map_location='cpu', weights_only=False)

def balanced(error, pos, neg):
    return 0.5 * ((error * pos).sum() / pos.sum().clamp_min(1) + (error * neg).sum() / neg.sum().clamp_min(1))

def sample_operator_rows(op, pixels):
    """Retain one interleaved pixel phase without averaging adjacent rays."""
    op = op.coalesce()
    index = op.indices()
    mapping = torch.full((op.shape[0],), -1, device=op.device, dtype=torch.long)
    mapping[pixels] = torch.arange(len(pixels), device=op.device)
    row = mapping[index[0]]
    keep = row >= 0
    return torch.sparse_coo_tensor(torch.stack([row[keep], index[1, keep]]), op.values()[keep], (len(pixels), op.shape[1])).coalesce()

def sample_permission(mask, h, w, pixels):
    return F.interpolate(mask.float()[None, None].cuda(), size=(h, w), mode='nearest')[0, 0].flatten()[pixels].bool()

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--finite-evidence', action='store_true')
    a.add_argument('--evidence-strength', choices=['confidence', 'positive_logit'], default='confidence')
    a.add_argument('--boundary-radius', type=int, default=1)
    a.add_argument('--heldout-stride', type=int, default=0, help='Every Nth source view is diagnostic only, excluded before fusion and fitting')
    a.add_argument('--conflict-abstention', action='store_true', help='Treat complement pixels supported by two other positive views as unknown')
    a.add_argument('--inputs', required=True, help='Explicit scene/response/evidence manifest; bypass all legacy query artifacts')
    a.add_argument('--key', required=True)
    a.add_argument('--source-fusion-only', action='store_true', help='Return confidence-weighted source backprojection without subset-render optimization')
    a.add_argument('--negative-likelihood', choices=['hard', 'contaminated', 'unknown', 'weighted'], default='hard')
    a.add_argument('--optimization-state-in')
    a.add_argument('--optimization-state-out')
    a.add_argument('--scope-state-out', help='Research-only export of fully assembled privileged query evidence')
    a.add_argument('--scope-state-only', action='store_true', help='Export scope evidence without fitting or producing a readout')
    a.add_argument('--pixel-loss', choices=['mean', 'balanced'], default='mean')
    a.add_argument('--source-coverage-reference')
    a.add_argument('--support-prior', choices=['none', 'count', 'footprint'], default='none')
    a.add_argument('--support-prior-weight', type=float, default=0.1)
    a.add_argument('--raster-scale', type=int, default=1)
    a.add_argument('--positive-witnesses', type=int, default=0)
    a.add_argument('--negative-admission', choices=['prior', 'local', 'complement', 'corroborated'], default='prior')
    a.add_argument('--positive-core', choices=['eroded', 'medial'], default='eroded')
    a.add_argument('--regional-positive-evidence', action='store_true')
    a.add_argument('--scope-initialization', choices=['semantic', 'source'], default='semantic')
    a.add_argument('--tracks-base', default=None)
    a.add_argument('--query-indices', type=int, nargs='*')
    a.add_argument('--extra-track-frames', action='store_true')
    a.add_argument('--retained-selection')
    a.add_argument('--semantic-base', default=None)
    a.add_argument('--cohort', choices=['old', 'coverage64', 'all'], default='all')
    a.add_argument('--evidence-base', default=None)
    a.add_argument('--output-base', required=True)
    a.add_argument('--steps', type=int, default=240)
    a.add_argument('--mode', choices=['admission', 'seeded'], required=True)
    args = a.parse_args()
    if args.scope_state_only and not args.scope_state_out:
        a.error('--scope-state-only requires --scope-state-out')
    args.carrier_arm = 'adaptive_allrgb'
    args.arm = 'decision'
    args.scope_negatives = False
    torch.set_num_threads(4)
    start = time.monotonic()
    inputs = json.loads(Path(args.inputs).read_text())
    scene = json.loads(Path(inputs['scene']).read_text())
    reliability_mode = scene.get('instance_evidence_reliability')
    local_reliability = reliability_mode in ('local', 'positive_local')
    response = load(inputs['response'])
    initial = dict(texts=response['texts'], scores=response['semantic_probability'], semantic_observed=response['observed'], selection_scores=response['semantic_probability'])
    prior = initial['selection_scores'].cuda()
    observed = initial['semantic_observed'].cuda()
    old_observed = observed
    exclusion_reference = prior
    binding = response['geometry']
    geometry_path = Path(binding['path'])
    geo = FrozenGaussianGeometry.load(geometry_path, expected_sha256=binding['sha256'], device='cuda')
    old_geo = geo
    geometry_prior = MogeGeometryPrior(scene['camera_report'], scene['camera_report_sha256'])
    source_frames = scene['source_frames']
    info = dict(train_frames=source_frames)
    (base_h, base_w) = inputs['base_raster']
    h = base_h * args.raster_scale
    w = base_w * args.raster_scale
    from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
    from types import SimpleNamespace
    records = []
    square = torch.zeros(prior.shape[1])
    count = torch.zeros_like(square)
    negative_admission_counts = torch.zeros(len(initial['texts']), dtype=torch.long)
    for frame in source_frames:
        d = load(Path(inputs['detections']) / f'identity_{frame}.pt')
        if d['queries'] != initial['texts']:
            raise ValueError('Evidence query order differs from explicit response')
        for r in d['instances']:
            if len(r['scores']):
                r['masks'] = F.interpolate(r['masks'].float()[:, None], size=(h, w), mode='bilinear', align_corners=False)[:, 0].half()
        records.append((frame, d))
    scale = (square / count.clamp_min(1)).sqrt().clamp_min(1e-06)
    views = []
    tracks = [load(path) if path else dict(seed_frame=None, frames={}) for path in inputs['tracks']]
    byframe = dict(records)
    seed_confidence = []
    for (qi, t) in enumerate(tracks):
        if t.get('seed_evidence') is not None:
            seed_confidence.append(t['seed_evidence']['seed_confidence'])
            continue
        seed_confidence.append(0.0)
        continue
        r = byframe[t['seed_frame']]['instances'][qi] if t['seed_frame'] is not None else None
        seed_confidence.append(float((r['scores'] * (r['foreground_margin'] / scale[qi]).sigmoid()).max()) if r is not None and len(r['scores']) else 0.0)
    for (view_index, (frame, d)) in enumerate(records):
        phase = view_index % args.raster_scale ** 2
        pixels = torch.arange(h * w, device='cuda').reshape(h, w)[phase // args.raster_scale::args.raster_scale, phase % args.raster_scale::args.raster_scale].flatten()
        camera = geometry_prior.cameras[frame]
        k = camera.intrinsic.cuda().clone()
        k[0] *= w / camera.width
        k[1] *= h / camera.height
        renderer = SimpleNamespace(scaled_intrinsics=lambda width, height: k, near_plane=0.001 * geometry_prior.scale, far_plane=100 * geometry_prior.scale)
        old_hit = rasterize_single_view_contributions(old_geo, renderer, camera.world_to_camera.cuda(), height=h, width=w, include_depths=False)
        old_hit['base_weights'] = old_hit['weights']
        (op_old, _, old_valid) = visible_operator(old_hit, old_observed, h * w, False)
        hit = rasterize_single_view_contributions(geo, renderer, camera.world_to_camera.cuda(), height=h, width=w, include_depths=False)
        hit['base_weights'] = hit['weights']
        (op, _, valid) = visible_operator(hit, observed, h * w, False)
        target = torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h, w) for r in d['instances']], -1).reshape(h * w, -1).cuda()
        detector = target.clone()
        present = torch.tensor([bool(t['frames'].get(frame, {}).get('present', False)) and frame != t['seed_frame'] for t in tracks], device='cuda')
        temporal = torch.stack([F.interpolate(t['frames'][frame]['probability'].float()[None, None], size=(h, w), mode='bilinear', align_corners=False)[0, 0] if bool(present[qi]) else torch.zeros(h, w) for (qi, t) in enumerate(tracks)], -1).reshape(h * w, -1).cuda()
        target = torch.maximum(target, temporal)
        if local_reliability:
            reliability = [instance_reliability(r, (h, w), prior.device,
                complement='view' if reliability_mode == 'positive_local' else 'minimum') for r in d['instances']]
            positive_weight = torch.stack([r[0] for r in reliability], -1).reshape(h * w, -1)
            negative_weight = torch.stack([r[1] for r in reliability])[None].expand(h * w, -1).clone()
            detector_positive_weight = positive_weight.clone()
            detector_negative_weight = negative_weight.clone()
            track_confidence = torch.tensor(seed_confidence, device='cuda') * present
            positive_weight = torch.where(temporal > detector, track_confidence[None], positive_weight)
            negative_weight = torch.where(present[None], torch.minimum(negative_weight, track_confidence[None]), negative_weight)
        binary = (target.T.reshape(-1, 1, h, w) > 0.5).float()
        positive_core = -F.max_pool2d(-binary, 2 * args.boundary_radius + 1, 1, args.boundary_radius) > 0.5
        if args.positive_core == 'medial':
            from scipy.ndimage import distance_transform_edt
            indices = args.query_indices if args.query_indices is not None else list(range(len(initial['texts'])))
            distance = torch.stack([torch.from_numpy(distance_transform_edt(binary[qi, 0].cpu().numpy()).copy()).float() for qi in indices])[:, None].cuda()
            ridge = (distance > 0) & (distance >= F.max_pool2d(distance, 3, 1, 1))
            positive_core[indices] |= ridge
        pos = positive_core.flatten(1).T & valid
        neg = (F.max_pool2d(binary, 2 * args.boundary_radius + 1, 1, args.boundary_radius) < 0.5).flatten(1).T & valid
        detector_binary = (detector.T.reshape(-1, 1, h, w) > 0.5).float()
        detector_pos = (-F.max_pool2d(-detector_binary, 2 * args.boundary_radius + 1, 1, args.boundary_radius) > 0.5).flatten(1).T & valid
        detector_neg = (F.max_pool2d(detector_binary, 2 * args.boundary_radius + 1, 1, args.boundary_radius) < 0.5).flatten(1).T & valid
        for (qi, r) in enumerate(d['instances']):
            rejected = r.get('unknown_masks')
            if rejected is not None and len(rejected):
                unknown = F.interpolate(rejected.float().amax(0)[None, None].cuda(), size=(h, w), mode='bilinear', align_corners=False)[0, 0].flatten() > 0.5
                neg[:, qi] &= ~unknown
                detector_neg[:, qi] &= ~unknown
        if args.raster_scale > 1:
            op = sample_operator_rows(op, pixels)
            op_old = sample_operator_rows(op_old, pixels)
            (pos, neg, target, detector, detector_pos, detector_neg, valid, old_valid) = [x[pixels] for x in (pos, neg, target, detector, detector_pos, detector_neg, valid, old_valid)]
            if local_reliability:
                positive_weight, negative_weight, detector_positive_weight, detector_negative_weight = [
                    x[pixels] for x in (positive_weight, negative_weight, detector_positive_weight, detector_negative_weight)]
        native_positive = _transport(op_old, exclusion_reference) > 0.5
        if args.arm in ('agreement', 'decision', 'anneal'):
            original_complement = neg
            neg = neg if args.scope_negatives else neg & old_valid & ~native_positive
            if args.negative_admission != 'prior':
                region = torch.tensor([t.get('seed_evidence') is not None for t in tracks], device='cuda')
                candidate = original_complement & old_valid & region[None]
                if args.negative_admission == 'local':
                    candidate &= (F.max_pool2d(binary, 9, 1, 4) > 0.5).flatten(1).T[pixels]
                negative_admission_counts += (candidate & ~neg).sum(0).cpu()
                if args.negative_admission != 'corroborated':
                    neg |= candidate
            if args.regional_positive_evidence:
                regional = torch.tensor([t.get('seed_evidence') is not None for t in tracks], device='cuda')
                pos = pos & (native_positive | regional[None])
        confidence = torch.tensor([float(r['scores'].max()) if len(r['scores']) else 0.0 for r in d['instances']], device='cuda')
        detector_confidence = confidence.clone()
        confidence = torch.maximum(confidence, torch.tensor(seed_confidence, device='cuda') * present)
        for (qi, t) in enumerate(tracks):
            evidence = t['frames'].get(frame, {})
            if 'positive_allowed' in evidence:
                pos[:, qi] &= sample_permission(evidence['positive_allowed'], h, w, pixels)
                neg[:, qi] &= sample_permission(evidence['negative_allowed'], h, w, pixels)
                if args.negative_admission == 'corroborated':
                    candidate[:, qi] &= sample_permission(evidence['negative_allowed'], h, w, pixels)
                confidence[qi] *= evidence['scope_reliability']
                if local_reliability:
                    positive_weight[:, qi] *= evidence['scope_reliability']
                    negative_weight[:, qi] *= evidence['scope_reliability']
        if not local_reliability:
            positive_weight = negative_weight = confidence[None]
            detector_positive_weight = detector_negative_weight = detector_confidence[None]
        camera = geometry_prior.cameras[frame]
        k = camera.intrinsic.cuda().clone()
        k[0] *= w / camera.width
        k[1] *= h / camera.height
        mass = torch.sparse.sum(op, dim=0).to_dense()
        views.append(dict(pixels=pixels, detector_pos=detector_pos, detector_neg=detector_neg, op=op, y=target, pos=pos, neg=neg, confidence=confidence, mass=mass, pending_negative=candidate if args.negative_admission == 'corroborated' else None, detector=detector, detector_confidence=detector_confidence, valid=valid, view=camera.world_to_camera.cuda()[None], k=k[None]))
        views[-1].update(positive_weight=positive_weight, negative_weight=negative_weight,
                         detector_positive_weight=detector_positive_weight, detector_negative_weight=detector_negative_weight)
    heldout_views = []
    if args.heldout_stride:
        heldout_views = [v for (i, v) in enumerate(views) if i % args.heldout_stride == 0]
        keep = [i for i in range(len(views)) if i % args.heldout_stride != 0]
        views = [views[i] for i in keep]
        records = [records[i] for i in keep]
    protected_negative_counts = torch.zeros(len(initial['texts']), dtype=torch.long)
    corroborated_negative_counts = torch.zeros(len(initial['texts']), dtype=torch.long)
    if args.negative_admission == 'corroborated':
        negative_mass = torch.zeros_like(prior)
        known_mass = torch.zeros_like(prior)
        positive_witness_count = torch.zeros_like(prior)
        for v in views:
            active = (v['confidence'] > 0) & v['pos'].any(0)
            if args.positive_witnesses:
                purity = _transport(v['op'].T, v['pos'].float()) / v['mass'][:, None].clamp_min(1e-08)
                v['positive_witness'] = (purity > 0.5) & active[None] & (v['mass'][:, None] > 1e-08)
                positive_witness_count += v['positive_witness']
            weight = v['confidence'][None] * active[None]
            known = (v['pos'] | v['pending_negative']).float() * weight
            negative = v['pending_negative'].float() * weight
            known_mass += _transport(v['op'].T, known)
            negative_mass += _transport(v['op'].T, negative)
        for v in views:
            active = (v['confidence'] > 0) & v['pos'].any(0)
            weight = v['confidence'][None] * active[None]
            known = (v['pos'] | v['pending_negative']).float() * weight
            negative = v['pending_negative'].float() * weight
            other_mass = known_mass - _transport(v['op'].T, known)
            other_negative = negative_mass - _transport(v['op'].T, negative)
            coverage = _transport(v['op'], (other_mass > 1e-08).float())
            fraction = _transport(v['op'], other_negative / other_mass.clamp_min(1e-08)) / coverage.clamp_min(1e-08)
            admitted = v['pending_negative'] & (coverage > 0.5) & (fraction > 0.5) & active[None]
            if args.positive_witnesses:
                other_witnesses = positive_witness_count - v['positive_witness'].float()
                protected = _transport(v['op'], (other_witnesses >= args.positive_witnesses).float()) > 0.5
                protected_negative_counts += (admitted & ~v['neg'] & protected).sum(0).cpu()
                admitted &= ~protected
            corroborated_negative_counts += (admitted & ~v['neg']).sum(0).cpu()
            v['neg'] |= admitted
        del negative_mass, known_mass, other_mass, other_negative
    previously_active = torch.stack([v['pos'].any(0) & (v['confidence'] > 0) for v in views]).any(0)
    total_mass = torch.zeros_like(prior)
    total_positive = torch.zeros_like(prior)
    for v in views:
        detector_pos = v['detector_pos']
        detector_neg = v['detector_neg']
        weight = v['detector_positive_weight'] * detector_pos + v['detector_negative_weight'] * detector_neg
        v['detector_pos'] = detector_pos
        v['detector_weight'] = weight
        total_mass += _transport(v['op'].T, weight)
        total_positive += _transport(v['op'].T, weight * detector_pos)
    admission = []
    for ((frame, _), v) in zip(records, views):
        m = _transport(v['op'].T, v['detector_weight'])
        n = _transport(v['op'].T, v['detector_weight'] * v['detector_pos'])
        other_mass = total_mass - m
        other = (total_positive - n) / other_mass.clamp_min(1e-08)
        coverage = _transport(v['op'], (other_mass > 1e-08).float())
        corroboration = _transport(v['op'], other) / coverage.clamp_min(1e-08)
        extra = v['detector_pos'] & (coverage > 0.5) & (corroboration > 0.5) & ~v['pos']
        for (qi, t) in enumerate(tracks):
            evidence = t['frames'].get(frame, {})
            if 'positive_allowed' in evidence:
                extra[:, qi] &= sample_permission(evidence['positive_allowed'], h, w, v['pixels'])
        v['pos'] |= extra
        v['neg'] &= ~extra
        admission.append(extra.sum(0).cpu())
    conflict_counts = torch.zeros(len(initial['texts']), dtype=torch.long)
    if args.conflict_abstention:
        witnesses = torch.zeros_like(prior)
        for v in views:
            active = (v['confidence'] > 0) & v['pos'].any(0)
            purity = _transport(v['op'].T, v['pos'].float()) / v['mass'][:, None].clamp_min(1e-08)
            v['positive_witness'] = (purity > 0.5) & active[None] & (v['mass'][:, None] > 1e-08)
            witnesses += v['positive_witness']
        for v in views:
            other = witnesses - v['positive_witness'].float()
            disputed = _transport(v['op'], (other >= 2).float()) > 0.5
            removed = v['neg'] & disputed
            conflict_counts += removed.sum(0).cpu()
            v['neg'] &= ~disputed
        del witnesses, other
    newly_active = torch.stack([v['pos'].any(0) & (v['confidence'] > 0) for v in views]).any(0) & ~previously_active
    if args.mode == 'seeded':
        seed = total_positive / total_mass.clamp_min(1e-08)
        prior[:, newly_active] = torch.maximum(prior[:, newly_active], seed[:, newly_active])
    admission_counts = torch.stack(admission).sum(0).tolist()
    if args.scope_state_out:
        torch.save(dict(texts=initial['texts'], source_frames=source_frames,
            source_raster=[h, w], privileged_query_evidence=True,
            views=[dict(target=v['y'].cpu(), positive=v['pos'].cpu(), negative=v['neg'].cpu(),
                        confidence=v['confidence'].cpu()) for v in views]), args.scope_state_out)
        if args.scope_state_only:
            return
    del total_mass, total_positive, m, n
    result = load(args.retained_selection)['selection_scores'].cuda() if args.retained_selection else prior.clone()
    history = []
    ones = torch.ones(len(prior), 1, device='cuda')
    coverage_reference = load(args.source_coverage_reference)['selection_scores'].cuda() if args.source_coverage_reference else None
    coverage_report = []
    optimization_state = load(args.optimization_state_in) if args.optimization_state_in else {}
    exported_state = {}
    steps = max(args.steps, len(source_frames))
    for (qi, text) in enumerate(initial['texts']):
        if args.query_indices is not None and qi not in args.query_indices:
            continue
        active = [v for v in views if v['confidence'][qi] > 0 and v['pos'][:, qi].any()]
        if args.optimization_state_in:
            fixed = optimization_state[qi]
            active = [views[i] for i in fixed['view_indices']]
            for (v, confidence, negative) in zip(active, fixed['confidence'], fixed['negative']):
                v['confidence'][qi] = confidence
                v['neg'][:, qi] = negative.cuda()
        if not active:
            history.append(dict(query=text, views=0, status='no_admitted_source_positive_keep_native_prior'))
            continue
        if local_reliability and scene.get('instance_evidence_mass') == 'matched':
            # Match the evidence actually consumed by this query's objective.
            # Detection-only views whose positive cores disappeared are not
            # fitted, so their complements cannot set the normalization budget.
            old_mass = sum(((v['pos'][:, qi] | v['neg'][:, qi]) * v['confidence'][qi]).sum() for v in active)
            local_mass = sum((v['pos'][:, qi] * v['positive_weight'][:, qi]
                             + v['neg'][:, qi] * v['negative_weight'][:, qi]).sum() for v in active)
            factor = old_mass / local_mass.clamp_min(1e-8)
            for v in active:
                v['positive_weight'][:, qi] *= factor
                v['negative_weight'][:, qi] *= factor
        if evidence_core.enabled():
            for v in active:
                domains = evidence_core.compile_domains(v['pos'][:, qi], v['neg'][:, qi],
                    visible=v['valid'][:, 0], negative_origin='model_complement')
                v['pos'][:, qi], v['neg'][:, qi] = domains.positive, domains.negative
        strength = source_strength([v['confidence'][qi] for v in active], args.evidence_strength)
        visibility = sum((v['mass'] for v in active))
        support = observed
        start_membership = prior[:, qi].clone()
        if args.scope_initialization == 'source':
            positive_mass = torch.zeros_like(start_membership)
            known_mass = torch.zeros_like(start_membership)
            for v in active:
                known = v['pos'][:, qi] * v['positive_weight'][:, qi] + v['neg'][:, qi] * v['negative_weight'][:, qi]
                known_mass += _transport(v['op'].T, known[:, None])[:, 0]
                positive_mass += _transport(v['op'].T, (known * v['y'][:, qi])[:, None])[:, 0]
            start_membership = torch.where(known_mass > 1e-08, positive_mass / known_mass.clamp_min(1e-08), start_membership)
        if args.source_fusion_only:
            membership = combine_with_native(start_membership, prior[:, qi], strength) if args.finite_evidence else start_membership
            result[:, qi] = torch.where(support, membership, prior[:, qi])
            history.append(dict(query=text, views=len(active), source_fusion_only=True, final_selected=int(((result[:, qi] > 0.5) & support).sum())))
            continue
        support_cost = torch.zeros_like(start_membership)
        if args.support_prior != 'none':
            from gsplat.cuda._wrapper import fully_fused_projection
            for v in active:
                (radii, _, _, conics, _) = fully_fused_projection(geo.xyz, None, geo.quaternions, geo.scales, v['view'], v['k'], w, h, near_plane=0.001 * geometry_prior.scale, far_plane=100 * geometry_prior.scale)
                seen = radii[0] > 0
                if args.support_prior == 'count':
                    support_cost += seen.float() * v['confidence'][qi]
                else:
                    c = conics[0, seen]
                    area = 2 * torch.pi / (c[:, 0] * c[:, 2] - c[:, 1].square()).clamp_min(1e-12).sqrt()
                    support_cost[seen] += area * geo.opacities.flatten()[seen] * v['confidence'][qi]
            support_cost *= support
            support_cost /= torch.dot(support_cost, start_membership).clamp_min(1e-08)
        if args.optimization_state_in:
            start_membership = fixed['start_membership'].cuda()
            support_cost = fixed['support_cost'].cuda()
        if args.optimization_state_out:
            exported_state[qi] = dict(view_indices=[i for (i, v) in enumerate(views) if any((v is a for a in active))], confidence=[float(v['confidence'][qi]) for v in active], negative=[v['neg'][:, qi].cpu() for v in active], start_membership=start_membership.cpu(), support_cost=support_cost.cpu())
        start_selected = int(((start_membership > 0.5) & support).sum())
        parameter = torch.nn.Parameter(torch.logit(start_membership.clamp(0.001, 0.999)))
        optimizer = torch.optim.Adam([parameter], lr=0.05)
        for step in range(steps):
            v = active[step % len(active)]
            proposal = parameter.sigmoid()
            membership = combine_with_native(proposal, prior[:, qi], strength) if args.finite_evidence else proposal
            p = torch.where(support, membership, prior[:, qi])
            response = _transport(v['op'], p[:, None])[:, 0]
            member = p * observed
            if args.arm in ('decision', 'anneal'):
                temperature = 0.5 * (0.1 / 0.5) ** (step / max(steps - 1, 1))
                logits = torch.logit(p.clamp(1e-06, 1 - 1e-06)) if args.finite_evidence else parameter
                member = torch.where(support, (logits / temperature).sigmoid(), (prior[:, qi] > 0.5).float()) * observed
            if args.mode in ('hard', 'hard_exposed'):
                member = (member > 0.5).float() + member - member.detach()
            (_, alpha, _) = rasterization(means=geo.xyz, quats=geo.quaternions, scales=geo.scales, opacities=geo.opacities.flatten() * member, colors=ones, viewmats=v['view'], Ks=v['k'], width=w, height=h, near_plane=0.001 * geometry_prior.scale, far_plane=100 * geometry_prior.scale, packed=False)
            silhouette = alpha[0, ..., 0].flatten()[v['pixels']]
            if args.arm == 'decision':
                cutoff = 10.5 / 255
                silhouette = torch.sigmoid(torch.logit(silhouette.clamp(1e-06, 1 - 1e-06)) - torch.logit(torch.tensor(cutoff, device='cuda')))
            known = v['pos'][:, qi] | v['neg'][:, qi]
            target = v['y'][:, qi]
            bce = F.binary_cross_entropy(silhouette.clamp(1e-05, 1 - 1e-05), target, reduction='none')
            if args.negative_likelihood == 'hard':
                effective_known = known
                pixel_loss = balanced(bce, v['pos'][:, qi], v['neg'][:, qi]) if args.pixel_loss == 'balanced' else (bce * known).sum() / known.sum().clamp_min(1)
                if local_reliability:
                    pw = v['positive_weight'][:, qi] / v['confidence'][qi].clamp_min(1e-8)
                    nw = v['negative_weight'][:, qi] / v['confidence'][qi].clamp_min(1e-8)
                    effective_known = v['pos'][:, qi] * pw + v['neg'][:, qi] * nw
                    pixel_loss = (.5 * ((bce * pw * v['pos'][:, qi]).sum() / v['pos'][:, qi].sum().clamp_min(1)
                                        + (bce * nw * v['neg'][:, qi]).sum() / v['neg'][:, qi].sum().clamp_min(1))
                                  if args.pixel_loss == 'balanced' else (bce * effective_known).sum() / known.sum().clamp_min(1))
            else:
                reliability = 0.5 if args.negative_likelihood == 'contaminated' else 0.0
                likelihood = 1 - reliability + reliability * torch.exp(-bce)
                negative_loss = -likelihood.log()
                posterior = (reliability * torch.exp(-bce) / likelihood).detach()
                if args.negative_likelihood == 'weighted':
                    negative_loss = 0.5 * bce
                    posterior = torch.full_like(bce, 0.5)
                pos = v['pos'][:, qi]
                neg = v['neg'][:, qi]
                effective_known = pos.float() + neg * posterior
                pixel_loss = 0.5 * ((bce * pos).sum() / pos.sum().clamp_min(1) + (negative_loss * neg).sum() / neg.sum().clamp_min(1))
            overlap = 1 - (2 * (silhouette * target * effective_known).sum() + 1) / ((silhouette * effective_known).sum() + (target * effective_known).sum() + 1)
            if local_reliability:
                overlap *= effective_known.sum() / known.sum().clamp_min(1)
            objective = v['confidence'][qi] * (pixel_loss + overlap)
            support_penalty = torch.dot(support_cost, p)
            objective += args.support_prior_weight * support_penalty
            loss_probability = balanced((response - target).square(), v['pos'][:, qi], v['neg'][:, qi])
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
        proposal = parameter.sigmoid().detach()
        membership = combine_with_native(proposal, prior[:, qi], strength) if args.finite_evidence else proposal
        result[:, qi] = torch.where(support, membership, prior[:, qi])
        history.append(dict(query=text, views=len(active), source_strength=float(strength), finite_evidence=args.finite_evidence, initial_selected=start_selected, final_selected=int(((result[:, qi] > 0.5) & support).sum()), final_support_cost=float(support_penalty.detach()), final_probability_loss=float(loss_probability.detach()), final_objective=float(objective.detach())))
        if coverage_reference is not None:
            with torch.inference_mode():
                comparisons = {}
                for (name, scores) in [('reference', coverage_reference[:, qi]), ('candidate', result[:, qi])]:
                    member = (scores > 0.5).float() * observed
                    counts = torch.zeros(4, device='cuda')
                    view_recall = []
                    for v in active:
                        (_, alpha, _) = rasterization(means=geo.xyz, quats=geo.quaternions, scales=geo.scales, opacities=geo.opacities.flatten() * member, colors=ones, viewmats=v['view'], Ks=v['k'], width=w, height=h, near_plane=0.001 * geometry_prior.scale, far_plane=100 * geometry_prior.scale, packed=False)
                        selected = alpha[0, ..., 0].flatten()[v['pixels']] > 10.5 / 255
                        pos = v['pos'][:, qi]
                        neg = v['neg'][:, qi]
                        counts += torch.stack([(selected & pos).sum(), pos.sum(), (selected & neg).sum(), neg.sum()])
                        view_recall.append(float((selected & pos).sum() / pos.sum().clamp_min(1)))
                    (tp, positive, fp, negative) = counts.tolist()
                    comparisons[name] = dict(positive_pixels=positive, negative_pixels=negative, positive_fraction=positive / max(1, positive + negative), positive_coverage=tp / max(1, positive), negative_false_positive_rate=fp / max(1, negative), mean_view_positive_coverage=sum(view_recall) / len(view_recall))
                coverage_report.append(dict(query=text, views=len(active), **comparisons))
        print(args.key, args.arm, qi, text, 'done', flush=True)
    heldout_report = []
    with torch.inference_mode():
        for (qi, text) in enumerate(initial['texts']):
            totals = torch.zeros(4, device='cuda')
            per_view = []
            for v in heldout_views:
                if not bool(v['confidence'][qi] > 0 and v['pos'][:, qi].any()):
                    continue
                member = (result[:, qi] > 0.5).float() * observed
                (_, alpha, _) = rasterization(means=geo.xyz, quats=geo.quaternions, scales=geo.scales, opacities=geo.opacities.flatten() * member, colors=ones, viewmats=v['view'], Ks=v['k'], width=w, height=h, near_plane=0.001 * geometry_prior.scale, far_plane=100 * geometry_prior.scale, packed=False)
                prediction = alpha[0, ..., 0].flatten()[v['pixels']] > 10.5 / 255
                positive = v['detector_pos'][:, qi]
                negative = v['detector_neg'][:, qi]
                tp = (prediction & positive).sum()
                fp = (prediction & negative).sum()
                totals += torch.stack([tp, positive.sum(), fp, negative.sum()])
                per_view.append(float(tp / (positive.sum() + fp).clamp_min(1)))
            (tp, pos, fp, neg) = totals.tolist()
            heldout_report.append(dict(query=text, views=len(per_view), positive_pixels=pos, negative_pixels=neg, positive_recall=tp / max(pos, 1), complement_false_positive_rate=fp / max(neg, 1), known_pixel_iou=tp / max(pos + fp, 1), mean_view_iou=sum(per_view) / max(len(per_view), 1)))
    if args.optimization_state_out:
        Path(args.optimization_state_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(exported_state, args.optimization_state_out)
    out = Path(args.output_base) / args.key
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(texts=initial['texts'], scores=initial['scores'], selection_scores=result.cpu(), selection_geometry=binding, selection_observed=observed.cpu(), semantic_geometry=binding, semantic_observed=observed.cpu()), out / 'elements.pt')
    torch.save(dict(texts=initial['texts'], scores=initial['scores'], selection_scores=result.cpu(), selection_geometry=binding, selection_observed=observed.cpu(), semantic_geometry=binding, semantic_observed=observed.cpu(), readout_contract='semantic_probability_and_independent_selected_set', semantic_parent=inputs['response']), out / 'typed_readout.pt')
    (out / 'fit.json').write_text(json.dumps(dict(evidence_strength=args.evidence_strength, finite_evidence=args.finite_evidence, native_reference_strength=1 if args.finite_evidence else 0, boundary_radius=args.boundary_radius, heldout_stride=args.heldout_stride, heldout_source_frames=[f for (i, f) in enumerate(source_frames) if args.heldout_stride and i % args.heldout_stride == 0], heldout_teacher_agreement=heldout_report, heldout_is_independent_ground_truth=False, conflict_abstention=args.conflict_abstention, conflict_negative_pixels_removed=conflict_counts.tolist(), explicit_inputs=args.inputs, implicit_legacy_query_artifacts_read=not bool(args.inputs), source_fusion_only=args.source_fusion_only, negative_likelihood=args.negative_likelihood, optimization_state_in=args.optimization_state_in, optimization_state_out=args.optimization_state_out, mode=args.mode, pixel_loss=args.pixel_loss, source_coverage_reference=args.source_coverage_reference, source_coverage=coverage_report, support_prior=args.support_prior, support_prior_weight=args.support_prior_weight, positive_witnesses=args.positive_witnesses, protected_negative_pixels=protected_negative_counts.tolist(), negative_admission=args.negative_admission, corroborated_negative_pixels=corroborated_negative_counts.tolist(), new_negative_pixels_before_confidence_and_track_gates=negative_admission_counts.tolist(), positive_core=args.positive_core, scope_initialization=args.scope_initialization, regional_positive_evidence=args.regional_positive_evidence, tracks_base=None, explicit_tracks=inputs['tracks'], query_indices=args.query_indices, extra_track_frames=args.extra_track_frames, retained_selection=args.retained_selection, steps=steps, source_raster=[h, w], sampled_raster=[base_h, base_w], interleaved_phases=args.raster_scale ** 2, seconds=time.monotonic() - start, geometry=str(geometry_path), source_evidence='explicit current-query SAM3 detections and declared tracks', source_frames=source_frames, minimum_requested_steps=args.steps, full_active_view_consumption=True, unknown_old_support_not_negative=True, semantic_base=None, hard_forward=args.mode in ('hard', 'hard_exposed'), optimize_all_observed=True, admitted_source_pixels=dict(zip(initial['texts'], admission_counts)), newly_active=[t for (t, b) in zip(initial['texts'], newly_active.tolist()) if b], semantic_scores_unchanged=True, target_rgb_used=False, history=history), indent=2) + '\n')
if __name__ == '__main__':
    main()
