"""Partial source-region relations supervise Gaussian members directly.

Per-view normalized evidence and a visibility witness prevent tiny occluded
contributions from becoming certain labels. Unknown stays unknown; conflicting
positive/negative evidence remains a soft target. No benchmark queries are read.
"""
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.sparse import diags
import torch
from torch.nn import functional as F

from .conditional_relation import enable
from .contracts import Support
from .relation_training import SourceRelations, buffer_digests, weighted_partial_loss
from .geometry import relaxed_subset_alpha
from .storage import load_field, save_field, sha256


def view_transfer(columns, observed, minimum_peak=.1):
    mass = np.asarray(columns.sum(0)).ravel()
    pixel_mass = np.asarray(columns.sum(1)).ravel()
    responsibility = diags(1 / np.maximum(pixel_mass, 1e-12)) @ columns
    peak = responsibility.max(axis=0).toarray().ravel()
    reliable = observed.numpy() & (peak >= minimum_peak) & (mass > 0)
    # Normalize within each visible Gaussian, not by its absolute opacity or
    # total raster contribution. Unreliable rows are excluded before division.
    inverse = np.zeros_like(mass); inverse[reliable] = 1 / mass[reliable]
    transfer = (diags(inverse) @ columns.T).tocsr()
    return transfer, peak.astype(np.float32)


class ElementRelations:
    def __init__(self, source, observed, cache_size=64):
        self.source = source; self.cache_size = cache_size; self.cache = OrderedDict()
        self.transfers, peaks = zip(*(view_transfer(v['columns'], observed.cpu()) for v in source.views))
        self.confidence = torch.from_numpy(np.mean(peaks, axis=0)).clamp(0, 1)
        self.confidence[~observed.cpu()] = 0

    def evidence(self, vi, ri):
        key = (vi, ri)
        if key not in self.cache:
            view = self.source.views[vi]
            raster = np.stack([view['positive'][ri].numpy(), view['negative'][ri].numpy()], axis=1).astype(np.float32)
            values = self.transfers[vi] @ raster
            rows = np.flatnonzero(values.sum(1) > 0)
            self.cache[key] = (rows, values[rows])
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]

    def combine(self, pair):
        parts = [(pair['primary_view'], pair['primary_region'], .5)]
        parts += [(pair['secondary_view'], ri, .5 * w) for ri, w in zip(pair['regions'], pair['weights'])]
        evidence = [(self.evidence(vi, ri), w) for vi, ri, w in parts]
        rows = np.unique(np.concatenate([x[0][0] for x in evidence]))
        values = np.zeros((len(rows), 2), dtype=np.float32)
        for (indices, value), weight in evidence:
            values[np.searchsorted(rows, indices)] += weight * value
        confidence = values.sum(1)
        target = np.divide(values[:, 0], confidence, out=np.zeros_like(confidence), where=confidence > 0)
        return torch.from_numpy(rows), torch.from_numpy(target), torch.from_numpy(confidence)

    def sample(self, pair, generator, per_class=128):
        rows, target, confidence = self.combine(pair)
        selected, classes = [], []
        for label, mask in enumerate([target < .5, target >= .5]):
            pool = torch.where(mask & (confidence > 0))[0]
            if len(pool):
                positions = pool[torch.randint(len(pool), (per_class,), generator=generator)]
                selected.append(positions); classes.append(torch.full((per_class,), label))
        if not selected:
            return None
        selected = torch.cat(selected)
        return rows[selected], target[selected], confidence[selected], torch.cat(classes)


def element_loss(logits, target, confidence, classes):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    terms = [(loss[classes == c] * confidence[classes == c]).sum() / confidence[classes == c].sum()
             for c in classes.unique()]
    return torch.stack(terms).mean()


def parent_logits(field, group, anchor, scale, rows):
    """Exact parent formula for this experiment's frozen metric/prototype core."""
    gate = torch.sigmoid(2 * (math.log(scale) - field.scale_cutoffs))
    prototype = F.normalize(F.normalize(group[anchor] * gate, dim=-1), dim=-1)
    cosine = F.normalize(group[rows] * gate, dim=-1) @ prototype
    distance = (field.xyz[rows] - field.xyz[anchor]) / (scale * field.world_units_per_metre)
    return field.log_temperature.exp().clamp(1, 50) * (cosine - field.affinity_bias) - .5 * distance.square().sum(-1)


@torch.no_grad()
def audit(field, original_group, observations, device, include_rendered=False):
    source = SourceRelations(observations, field.observed, split='development')
    elements = ElementRelations(source, field.observed)
    generator = torch.Generator().manual_seed(20261011)
    sample_generator = torch.Generator().manual_seed(20261012)
    values = dict(parent_bce=[], refined_bce=[])
    if include_rendered:
        values.update(parent_rendered_bce=[], refined_rendered_bce=[])
    skipped = 0
    for i in range(len(source.examples)):
        pair = source.pair(i, generator); sample = elements.sample(pair, sample_generator)
        if sample is None:
            skipped += 1; continue
        rows, target, confidence, classes = [x.to(device) for x in sample]
        support = Support(torch.tensor([pair['anchor']], device=device), torch.ones(1, device=device))
        before = parent_logits(field, original_group, pair['anchor'], pair['scale'], rows)
        after = field.member_logits(support, pair['scale'], rows)
        for key, logits in [('parent_bce', before), ('refined_bce', after)]:
            values[key].append(float(element_loss(logits, target, confidence, classes)))
        if include_rendered:
            all_rows = torch.arange(len(field.xyz), device=device)
            members = dict(parent=parent_logits(field, original_group, pair['anchor'], pair['scale'], all_rows).sigmoid(),
                           refined=field.support_membership(support, pair['scale']))
            primary = source.views[pair['primary_view']]
            secondary = source.views[pair['secondary_view']]
            ri = pair['primary_region']; height, width = source.manifest['raster_hw']
            for name, member in members.items():
                losses = []
                for view, positive, negative in [(primary, primary['positive'][ri], primary['negative'][ri]),
                                                 (secondary, pair['positive'], pair['negative'])]:
                    rendered = relaxed_subset_alpha(field, member, view['camera'], height, width,
                                                    source.manifest['world_units_per_metre'])
                    losses.append(float(weighted_partial_loss(rendered, positive.to(device).float(), negative.to(device).float())))
                values[name + '_rendered_bce'].append(float(np.mean(losses)))
    if not values['parent_bce']:
        raise ValueError('no held-out source element relations')
    return dict(frames=[v['frame'] for v in source.views], regions=len(source.examples), skipped=skipped,
                metrics={k: float(np.mean(v)) for k, v in values.items()},
                used_for_optimization=False, used_for_selection=False)


def fit(parent, observations, output, *, arm, epochs=2, seed=20261008, learning_rate=.001,
        per_class=128, device='cuda'):
    if arm not in {'element_relation', 'conditional_relation', 'joint_conditional_relation'}:
        raise ValueError('unknown element relation arm')
    rendered_weight = .5 if arm == 'joint_conditional_relation' else 0.
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); field, manifest = load_field(parent, device)
    source_hash = sha256(Path(observations) / 'observations.json')
    if manifest['provenance']['observations_sha256'] != source_hash:
        raise ValueError('source relation parent differs')
    if (field.local_metric, field.spatial_metric, field.support_aggregation, field.negative_rule,
        field.relation_decoder) != ('none', 'scale_normalized', 'normalized_prototype', 'product', 'none'):
        raise ValueError('this controlled study requires the declared parent member formula')
    original_group = field.group.detach().clone()
    source = SourceRelations(observations, field.observed)
    if source.manifest['world_units_per_metre'] != field.world_units_per_metre:
        raise ValueError('source geometry units differ')
    elements = ElementRelations(source, field.observed)
    if arm != 'element_relation':
        enable(field, elements.confidence, seed)
    before = buffer_digests(field)
    for name, parameter in field.named_parameters():
        parameter.requires_grad_(name == 'group' or name.startswith('relation.'))
    protected = {name: p.detach().clone() for name, p in field.named_parameters() if not p.requires_grad}
    parameters = [p for p in field.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    sample_generator = torch.Generator().manual_seed(seed + 1)
    visits = torch.zeros(len(source.examples), dtype=torch.long)
    exposure = torch.zeros(len(field.xyz), dtype=torch.long)
    trace, history, skipped = [], [], 0
    sample_digest = hashlib.sha256()
    sample_count, soft_count, confidence_sum = 0, 0, 0.
    if rendered_weight:
        render_gradient_count = torch.zeros(len(field.xyz), dtype=torch.int32, device=device)
        render_gradient_mass = torch.zeros(len(field.xyz), device=device)
        height, width = source.manifest['raster_hw']
    for epoch in range(epochs):
        for example in torch.randperm(len(source.examples), generator=generator).tolist():
            pair = source.pair(example, generator); visits[example] += 1
            primary = source.views[pair['primary_view']]; secondary = source.views[pair['secondary_view']]
            row = {k: v for k, v in pair.items() if k not in {'positive', 'negative'}}
            row.update(primary_frame=primary['frame'], secondary_frame=secondary['frame'], epoch=epoch,
                example=example, disagreement_pixels=int(((pair['positive'] > 0) & (pair['negative'] > 0)).sum()))
            trace.append(row)
            sample = elements.sample(pair, sample_generator, per_class)
            if sample is None:
                skipped += 1; continue
            for value in sample:
                sample_digest.update(memoryview(value.contiguous().numpy()))
            sample_count += len(sample[0])
            soft_count += int(((sample[1] > .05) & (sample[1] < .95)).sum())
            confidence_sum += float(sample[2].sum())
            exposure.scatter_add_(0, sample[0], torch.ones_like(sample[0]))
            rows, target, confidence, classes = [x.to(device) for x in sample]
            support = Support(torch.tensor([pair['anchor']], device=device), torch.ones(1, device=device))
            optimizer.zero_grad(set_to_none=True)
            logits = field.member_logits(support, pair['scale'], rows)
            loss = element_loss(logits, target, confidence, classes)
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite element relation loss')
            ((1 - rendered_weight) * loss).backward()
            element_value = float(loss.detach()); render_values = []
            if rendered_weight:
                ri = pair['primary_region']
                for view, positive, negative in [(primary, primary['positive'][ri], primary['negative'][ri]),
                                                 (secondary, pair['positive'], pair['negative'])]:
                    full_logits = field.support_logits(support, pair['scale'])
                    full_logits.retain_grad()
                    rendered = relaxed_subset_alpha(field, full_logits.sigmoid(), view['camera'], height, width,
                                                    source.manifest['world_units_per_metre'])
                    rendered_loss = weighted_partial_loss(rendered, positive.to(device).float(), negative.to(device).float())
                    if not bool(torch.isfinite(rendered_loss)):
                        raise ValueError('nonfinite joint rendered relation loss')
                    (rendered_weight * .5 * rendered_loss).backward()
                    gradient = full_logits.grad.detach()
                    render_gradient_count.add_(gradient.ne(0))
                    render_gradient_mass.add_(gradient.abs())
                    render_values.append(float(rendered_loss.detach()))
                    del full_logits, rendered, rendered_loss, gradient
            torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
            if len(trace) % 100 == 0 or len(trace) == epochs * len(source.examples):
                status = dict(step=len(trace), steps=epochs * len(source.examples), loss=element_value)
                if rendered_weight:
                    status.update(element_loss=element_value, rendered_loss=float(np.mean(render_values)),
                                  loss=(1-rendered_weight)*element_value + rendered_weight*float(np.mean(render_values)))
                history.append(status); print(arm, status, flush=True)
    if buffer_digests(field) != before or any(not torch.equal(p, dict(field.named_parameters())[k]) for k, p in protected.items()):
        raise ValueError('element training changed protected state')
    if not bool((visits == epochs).all()):
        raise ValueError('incomplete source region exposure')
    # Release training-only projection targets before opening held-out views.
    del elements
    development = audit(field, original_group, observations, device, include_rendered=bool(rendered_weight))
    export = save_field(field, output / 'field', dict(parent_field_sha256=manifest['files']['field.pt']['sha256'],
        observations_sha256=source_hash, supervision=('source-only joint partial element and subset-alpha relations'
            if rendered_weight else 'source-only partial per-Gaussian region relations'),
        arm=arm, additional_3d_labels=False, benchmark_queries_used=False))
    observed = field.observed.cpu(); counts = exposure[observed]
    report = dict(arm=arm, parent=str(parent), parent_sha256=manifest['files']['field.pt']['sha256'],
        observations=str(observations), observations_sha256=source_hash, epochs=epochs, seed=seed,
        steps=len(trace), skipped_steps=skipped, learning_rate=learning_rate, per_class=per_class,
        source_frames=[v['frame'] for v in source.views], training_regions=len(source.examples),
        each_region_visits=epochs, source_development=development, trace=trace, history=history,
        sample_trace_sha256=sample_digest.hexdigest(), sampled_elements=sample_count,
        ambiguous_sample_fraction=soft_count / max(sample_count, 1),
        mean_sample_known_confidence=confidence_sum / max(sample_count, 1),
        row_exposure=dict(sampled_rows=int((counts > 0).sum()), observed_rows=len(counts),
                          sample_quantiles=torch.quantile(counts.float(), torch.tensor([0., .25, .5, .75, 1.])).tolist()),
        shared_parameters_protected=list(protected), protected_state_unchanged=True,
        export_sha256=export['files']['field.pt']['sha256'], seconds=time.monotonic() - started,
        source_access_at_query=False, benchmark_queries_used=False)
    if rendered_weight:
        counts = render_gradient_count[field.observed].float()
        mass = render_gradient_mass[field.observed]
        if not bool(torch.isfinite(mass).all()):
            raise ValueError('nonfinite rendered logit gradient accounting')
        report.update(rendered_weight=rendered_weight, element_weight=1-rendered_weight,
            rendered_logit_gradient=dict(observed_rows=len(counts), nonzero_rows=int((counts > 0).sum()),
                nonzero_observed_fraction=float((counts > 0).float().mean()),
                count_quantiles=torch.quantile(counts, torch.tensor([0., .25, .5, .75, 1.], device=device)).tolist(),
                absolute_gradient_total=float(mass.sum()),
                absolute_gradient_quantiles=torch.quantile(mass, torch.tensor([0., .25, .5, .75, 1.], device=device)).tolist(),
                definition='actual weighted rendered-loss gradients at full member logits before parameter clipping; not parameter-update coverage'))
    (output / 'training.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
