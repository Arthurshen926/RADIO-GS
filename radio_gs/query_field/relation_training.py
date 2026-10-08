"""Source-only, view-balanced partial relation supervision on one Gaussian field.

Source masks and projection columns are construction artifacts. The deployed
model is the existing standalone CapabilityField with no observation sidecar.
"""
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csc_matrix
import torch

from .contracts import Support
from .geometry import relaxed_subset_alpha
from .observations import training_episode
from .storage import load_field, save_field, sha256


def weighted_partial_loss(predicted, positive, negative):
    """View-balanced soft partial evidence; zero/zero is unknown, not background."""
    terms = []
    for mass, probability in [(positive, predicted), (negative, 1 - predicted)]:
        if bool(mass.sum() > 0):
            terms.append(-(probability.clamp_min(1e-6).log() * mass).sum() / mass.sum())
    if not terms:
        raise ValueError('partial relation has no known evidence')
    return torch.stack(terms).mean()


def compatible_regions(interiors, scales, pixels, weights, scale):
    """All scale-compatible masks with majority interior projection support.

    No visible projection means no evidence. Multiple nested masks retain
    separate votes before within-view normalization; they do not create views.
    """
    if not len(pixels) or not bool(weights.sum() > 0):
        return torch.empty(0, dtype=torch.long), torch.empty(0)
    coverage = interiors[:, pixels].float() @ (weights / weights.sum())
    ratio = scales / float(scale)
    valid = (coverage >= .5) & (ratio >= .5) & (ratio <= 2)
    indices = torch.where(valid)[0]
    mass = coverage[indices]
    return indices, mass / mass.sum() if len(mass) else mass


class SourceRelations:
    def __init__(self, root, observed, split='train'):
        if split not in {'train', 'development'}:
            raise ValueError('unknown relation source split')
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'observations.json').read_text())
        splits = self.manifest['splits']
        if set(splits['train_frames']) & set(splits['development_frames']):
            raise ValueError('source training/development views overlap')
        self.views, self.examples = [], []
        observed = observed.cpu()
        for frame in splits[split + '_frames']:
            entry = next(x for x in self.manifest['frames'] if x['frame'] == frame)
            path = self.root / f'view_{frame:05d}.pt'
            if entry['split'] != split or sha256(path) != entry['compiled']['sha256']:
                raise ValueError('source relation observation or split changed')
            raw = torch.load(path, map_location='cpu', weights_only=True)
            operator = raw['operator'].coalesce()
            pixels, rows = operator.indices().numpy()
            # Geometric contribution, not the row-normalized semantic operator.
            values = operator.values().numpy() * raw['raw_mass'].numpy()[pixels]
            columns = csc_matrix((values, (pixels, rows)), shape=operator.shape)
            positive, negative, supports, indices = [], [], [], []
            for mask_index in range(len(raw['masks'])):
                episode = training_episode(raw, mask_index, observed)
                if episode is None:
                    continue
                pos, neg, support_pixels = episode
                positive.append(pos); negative.append(neg); supports.append(support_pixels)
                indices.append(mask_index)
            if not positive:
                continue
            view = dict(frame=frame, columns=columns, positive=torch.stack(positive),
                negative=torch.stack(negative), pixels=supports, best=raw['best_element'],
                scales=raw['scales_m'][indices], original_regions=indices,
                camera=SimpleNamespace(intrinsic=raw['intrinsic'], world_to_camera=raw['world_to_camera'],
                    height=raw['original_hw'][0], width=raw['original_hw'][1]))
            self.examples.extend((len(self.views), i) for i in range(len(indices)))
            self.views.append(view)
        if not self.examples:
            raise ValueError('no valid source relations')

    def association(self, view, anchor, scale):
        columns = view['columns']
        start, stop = columns.indptr[anchor:anchor + 2]
        pixels = torch.from_numpy(columns.indices[start:stop].astype(np.int64))
        mass = torch.from_numpy(columns.data[start:stop])
        return compatible_regions(view['positive'], view['scales'], pixels, mass, scale)

    def pair(self, example, generator):
        vi, ri = self.examples[example]
        primary = self.views[vi]
        pixels = primary['pixels'][ri]
        pixel = pixels[int(torch.randint(len(pixels), (1,), generator=generator))]
        anchor = int(primary['best'][pixel])
        scale = float(primary['scales'][ri])
        options = []
        for oi, other in enumerate(self.views):
            if oi == vi:
                continue
            indices, weights = self.association(other, anchor, scale)
            if len(indices):
                options.append((oi, indices, weights))
        base = dict(primary_view=vi, primary_region=ri, anchor=anchor, scale=scale)
        if not options:
            return dict(**base, secondary_view=vi, secondary_anchor=anchor,
                positive=primary['positive'][ri].float(), negative=primary['negative'][ri].float(),
                regions=[ri], weights=[1.], cross_view=False, jointly_supported_views=0)
        oi, indices, weights = options[int(torch.randint(len(options), (1,), generator=generator))]
        other = self.views[oi]
        pos = weights @ other['positive'][indices].float()
        neg = weights @ other['negative'][indices].float()
        selected_region = int(indices[weights.argmax()])
        support_pixels = other['pixels'][selected_region]
        # A local supervised control sees the exact same second-view evidence.
        # Only the member condition differs: its own interior vs the first anchor.
        secondary = int(other['best'][support_pixels[len(support_pixels) // 2]])
        return dict(**base, secondary_view=oi, secondary_anchor=secondary, positive=pos, negative=neg,
                    regions=indices.tolist(), weights=weights.tolist(), cross_view=True,
                    jointly_supported_views=len(options))


def buffer_digests(field):
    result = {}
    for key, value in field.named_buffers():
        h = hashlib.sha256()
        for part in value.detach().reshape(-1).split(1024 * 1024):
            h.update(memoryview(part.cpu().contiguous().numpy()))
        result[key] = h.hexdigest()
    return result


def training_parameters(field, policy):
    """Separate Gaussian relation learning from shared acceptance calibration."""
    if policy not in {'joint', 'group_only'}:
        raise ValueError('unknown relation parameter policy')
    for name, parameter in field.named_parameters():
        parameter.requires_grad_(policy == 'joint' or name == 'group')
    protected = {name: p.detach().clone() for name, p in field.named_parameters() if not p.requires_grad}
    return [p for p in field.parameters() if p.requires_grad], protected


def fit(parent, observations, output, *, arm, epochs=2, seed=20261008, learning_rate=.001,
        parameter_policy='joint', device='cuda'):
    if arm not in {'local_anchor_control', 'cross_view_anchor'}:
        raise ValueError('unknown relation training arm')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    field, parent_manifest = load_field(parent, device)
    before = buffer_digests(field)
    source = SourceRelations(observations, field.observed)
    if source.manifest['world_units_per_metre'] != field.world_units_per_metre:
        raise ValueError('source/field geometry units differ')
    # Bind source rows to the exact construction used by the parent checkpoint.
    if parent_manifest['provenance']['observations_sha256'] != sha256(Path(observations) / 'observations.json'):
        raise ValueError('parent and relation observations differ')
    generator = torch.Generator().manual_seed(seed)
    parameters, protected = training_parameters(field, parameter_policy)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    history, trace = [], []
    visits = torch.zeros(len(source.examples), dtype=torch.long)
    height, width = source.manifest['raster_hw']
    nsteps = epochs * len(source.examples)
    for epoch in range(epochs):
        for example in torch.randperm(len(source.examples), generator=generator).tolist():
            pair = source.pair(example, generator)
            visits[example] += 1
            primary = source.views[pair['primary_view']]
            secondary = source.views[pair['secondary_view']]
            ri = pair['primary_region']
            optimizer.zero_grad(set_to_none=True)
            losses = []
            secondary_anchor = pair['anchor'] if arm == 'cross_view_anchor' else pair['secondary_anchor']
            conditions = [(pair['anchor'], primary, primary['positive'][ri], primary['negative'][ri]),
                          (secondary_anchor, secondary, pair['positive'], pair['negative'])]
            for anchor, view, positive, negative in conditions:
                support = Support(torch.tensor([anchor], device=device), torch.ones(1, device=device))
                member = field.support_membership(support, pair['scale'])
                rendered = relaxed_subset_alpha(field, member, view['camera'], height, width,
                                                 source.manifest['world_units_per_metre'])
                loss = weighted_partial_loss(rendered, positive.to(device).float(), negative.to(device).float())
                if not bool(torch.isfinite(loss)):
                    raise ValueError('nonfinite relation loss')
                (.5 * loss).backward()
                losses.append(float(loss.detach()))
                del member, rendered, loss
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            row = {k: v for k, v in pair.items() if k not in {'positive', 'negative'}}
            row.update(primary_frame=primary['frame'], secondary_frame=secondary['frame'],
                       epoch=epoch, example=example,
                       disagreement_pixels=int(((pair['positive'] > 0) & (pair['negative'] > 0)).sum()))
            trace.append(row)
            step = len(trace)
            if step % 100 == 0 or step == nsteps:
                status = dict(step=step, steps=nsteps, loss=sum(losses) / 2,
                              primary_loss=losses[0], secondary_loss=losses[1])
                history.append(status)
                print(arm, status, flush=True)
    if buffer_digests(field) != before:
        raise ValueError('relation training modified protected geometry/native/support buffers')
    if any(not torch.equal(value, dict(field.named_parameters())[name]) for name, value in protected.items()):
        raise ValueError('relation training modified protected shared member parameters')
    if not bool((visits == epochs).all()):
        raise ValueError('source region exposure is incomplete')
    export = save_field(field, output / 'field', dict(parent_field_sha256=parent_manifest['files']['field.pt']['sha256'],
        observations_sha256=sha256(Path(observations) / 'observations.json'),
        supervision='source-only partial multi-view SAM relations', arm=arm, parameter_policy=parameter_policy,
        additional_3d_labels=False, benchmark_queries_used=False))
    report = dict(arm=arm, parent=str(parent), parent_sha256=parent_manifest['files']['field.pt']['sha256'],
        observations=str(observations), observations_sha256=sha256(Path(observations) / 'observations.json'),
        epochs=epochs, steps=nsteps, learning_rate=learning_rate, seed=seed,
        parameter_policy=parameter_policy, protected_member_parameters=list(protected),
        protected_member_parameters_unchanged=True,
        source_frames=[v['frame'] for v in source.views], training_regions=len(source.examples),
        each_region_visits=epochs, rendered_losses=2 * nsteps, protected_buffers_unchanged=True,
        cross_view_steps=sum(t['cross_view'] for t in trace),
        distinct_anchor_steps=sum(t['anchor'] != t['secondary_anchor'] for t in trace),
        ambiguity_steps=sum(t['disagreement_pixels'] > 0 for t in trace),
        trace=trace, history=history, seconds=time.monotonic() - started,
        export_sha256=export['files']['field.pt']['sha256'], source_access_at_query=False)
    (output / 'training.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
