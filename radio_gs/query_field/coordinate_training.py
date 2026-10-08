"""Direct full-native Gaussian coordinate learning from source query functions.

CPU-resident coordinates and row-local Adam moments preserve full capacity. Only
the sampled visible rows participate in each GPU step; this is sparse parameter
optimization, not a lower-dimensional codec or an additional runtime memory.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .functional_training import FunctionalSource, preference_loss, projected_query_scores, unchanged_state
from .storage import load_field, save_field, sha256


class CoordinateAdam:
    """Adam with a separate update count per observed row; absent rows stay exact."""
    def __init__(self, codes, learning_rate=.0001):
        self.values = codes.float().clone()
        self.first = torch.zeros_like(self.values)
        self.second = torch.zeros_like(self.values)
        self.steps = torch.zeros(len(codes), dtype=torch.long)
        self.learning_rate = learning_rate

    @torch.no_grad()
    def update(self, rows, gradient):
        if rows.ndim != 1 or len(rows.unique()) != len(rows):
            raise ValueError('coordinate updates require unique rows')
        gradient = gradient.detach().cpu().float()
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError('nonfinite coordinate gradient')
        self.steps[rows] += 1
        first = .9 * self.first[rows] + .1 * gradient
        second = .999 * self.second[rows] + .001 * gradient.square()
        self.first[rows] = first; self.second[rows] = second
        steps = self.steps[rows].float()[:, None]
        corrected_first = first / (1 - torch.pow(.9, steps))
        corrected_second = second / (1 - torch.pow(.999, steps))
        self.values[rows] -= self.learning_rate * corrected_first / (corrected_second.sqrt() + 1e-8)

    def release_moments(self):
        self.first = self.second = None


@torch.no_grad()
def source_audit(field, coordinates, source, device):
    generator = torch.Generator().manual_seed(20261010)
    values = {c: dict(parent_kl=[], refined_kl=[]) for c in coordinates}
    for index in range(len(source.examples)):
        rows, operator, queries, teacher = source.sample(index, generator, device=device)
        for channel, state in coordinates.items():
            for name, codes in [('parent', getattr(field, channel).codes[rows]),
                                ('refined', state.values[rows])]:
                score = projected_query_scores(codes.to(device).float(), queries[channel], operator)
                values[channel][name + '_kl'].append(float(preference_loss(score, teacher[channel])))
    return dict(regions=len(source.examples), frames=[v['frame'] for v in source.views], seed=20261010,
        metrics={c: {k: float(np.mean(v)) for k, v in row.items()} for c, row in values.items()},
        used_for_optimization=False, used_for_selection=False)


def fit(parent, observations, output, *, epochs=2, seed=20261009, learning_rate=.0001,
        pairs=64, device='cuda'):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    field, manifest = load_field(parent)
    source_digest = sha256(Path(observations) / 'observations.json')
    if manifest['provenance']['observations_sha256'] != source_digest:
        raise ValueError('coordinate source and parent differ')
    if any(getattr(field, c).weight is not None for c in ['semantic', 'local']):
        raise ValueError('coordinate control requires the full-native parent')
    before, parameters = unchanged_state(field)
    source = FunctionalSource(observations, field.observed)
    if source.manifest['world_units_per_metre'] != field.world_units_per_metre:
        raise ValueError('source/field world units differ')
    coordinates = {c: CoordinateAdam(getattr(field, c).codes, learning_rate) for c in ['semantic', 'local']}
    generator = torch.Generator().manual_seed(seed)
    visits = torch.zeros(len(source.examples), dtype=torch.long)
    step, steps, history = 0, epochs * len(source.examples), []
    for epoch in range(epochs):
        for index in torch.randperm(len(source.examples), generator=generator).tolist():
            rows, operator, queries, teacher = source.sample(index, generator, pairs, device)
            active, losses = {}, {}
            for channel, state in coordinates.items():
                active[channel] = state.values[rows].to(device).detach().requires_grad_()
                original = getattr(field, channel).codes[rows].to(device).float()
                decoded = F.normalize(active[channel], dim=-1)
                score = projected_query_scores(decoded, queries[channel], operator)
                functional = preference_loss(score, teacher[channel])
                trust = (1 - (decoded * F.normalize(original, dim=-1)).sum(-1)).mean()
                loss = .5 * (functional + trust)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('nonfinite coordinate objective')
                loss.backward()
                losses[channel] = dict(functional=float(functional.detach()), trust=float(trust.detach()))
            torch.nn.utils.clip_grad_norm_(list(active.values()), 1.)
            for channel, state in coordinates.items():
                state.update(rows, active[channel].grad)
            visits[index] += 1; step += 1
            if step % 100 == 0 or step == steps:
                row = dict(step=step, steps=steps, losses=losses)
                history.append(row); print('native_coordinates', row, flush=True)
    if not bool((visits == epochs).all()):
        raise ValueError('incomplete coordinate source exposure')
    exposure = {}
    for channel, state in coordinates.items():
        counts = state.steps[field.observed]
        exposure[channel] = dict(rows_updated=int((counts > 0).sum()), observed_rows=len(counts),
            updates_quantiles=torch.quantile(counts.float(), torch.tensor([0., .25, .5, .75, 1.])).tolist())
        state.release_moments()
    development = FunctionalSource(observations, field.observed, 'development')
    audit = source_audit(field, coordinates, development, device)
    drift = {}
    with torch.no_grad():
        for channel, state in coordinates.items():
            codes = getattr(field, channel).codes
            error, count = 0., 0
            for start in range(0, len(codes), 8192):
                stop = min(start + 8192, len(codes)); eligible = field.observed[start:stop]
                original = codes[start:stop][eligible].to(device).float()
                decoded = F.normalize(state.values[start:stop][eligible].to(device), dim=-1)
                if not bool(torch.isfinite(decoded).all()):
                    raise ValueError('nonfinite baked coordinate')
                error += float((1 - (F.normalize(original, dim=-1) * decoded).sum(-1)).sum())
                count += len(decoded)
                codes[start:stop][eligible] = decoded.cpu().to(codes.dtype)
            drift[channel] = error / max(count, 1)
    after, _ = unchanged_state(field)
    if after != before or any(not torch.equal(p, dict(field.named_parameters())[k]) for k, p in parameters.items()):
        raise ValueError('coordinate fit changed protected field state')
    export = save_field(field, output / 'field', dict(parent_field_sha256=manifest['files']['field.pt']['sha256'],
        observations_sha256=source_digest, supervision='source-only native pair preferences',
        construction='direct full-native Gaussian coordinate optimization',
        benchmark_queries_used=False, additional_3d_labels=False))
    report = dict(parent=str(parent), parent_sha256=manifest['files']['field.pt']['sha256'],
        observations=str(observations), observations_sha256=source_digest, epochs=epochs, seed=seed,
        learning_rate=learning_rate, pairs=pairs, temperature=.1, trust_weight=1.,
        optimizer='per-row Adam, beta=(.9,.999), eps=1e-8, no weight decay',
        steps=steps, training_regions=len(source.examples), each_region_visits=epochs,
        source_frames=[v['frame'] for v in source.views], history=history, source_development=audit,
        row_exposure=exposure, mean_direction_drift=drift, protected_state_unchanged=True,
        export_sha256=export['files']['field.pt']['sha256'], seconds=time.monotonic() - started,
        source_access_at_query=False, benchmark_queries_used=False)
    (output / 'training.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
