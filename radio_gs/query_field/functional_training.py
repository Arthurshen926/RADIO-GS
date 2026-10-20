"""Source-only native query preservation, baked into one Gaussian field.

The full-dimensional residual maps are construction parameters, not additional
scene memories. Teacher preferences supervise projected native *query scores*.
SAM defines reliable comparison domains, never hard DINO similarity targets.
"""
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix, diags
import torch
from torch import nn
from torch.nn import functional as F

from .observations import training_episode
from .storage import load_field, save_field, sha256, field_identity, deployment_bytes


class NativeResidual(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(dimension, dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, codes):
        original = F.normalize(codes.float(), dim=-1)
        return F.normalize(original + F.linear(original, self.delta, self.bias), dim=-1)


def preference_loss(student, teacher, temperature=.1):
    """KL of paired teacher preferences; both vectors are [positive, negative]."""
    a, b = student.chunk(2)
    ta, tb = teacher.detach().chunk(2)
    target_logit = (ta - tb) / temperature
    target = target_logit.sigmoid()
    return (F.binary_cross_entropy_with_logits((a - b) / temperature, target)
            - F.binary_cross_entropy_with_logits(target_logit, target))


def projected_query_scores(codes, query, operator):
    # Normalize per Gaussian before observation integration. Normalizing the
    # projected feature mean would train a different deployed query function.
    values = F.normalize(codes.float(), dim=-1) @ F.normalize(query.float(), dim=-1)
    return torch.sparse.mm(operator, values[:, None]).flatten()


class FunctionalSource:
    def __init__(self, root, observed, split='train'):
        if split not in {'train', 'development'}:
            raise ValueError('unknown source split')
        root = Path(root)
        self.manifest = json.loads((root / 'observations.json').read_text())
        splits = self.manifest['splits']
        if set(splits['train_frames']) & set(splits['development_frames']):
            raise ValueError('source training/development overlap')
        observed = observed.cpu()
        self.views, self.examples = [], []
        for frame in splits[split + '_frames']:
            entry = next(e for e in self.manifest['frames'] if e['frame'] == frame)
            path = root / f'view_{frame:05d}.pt'
            if entry['split'] != split or sha256(path) != entry['compiled']['sha256']:
                raise ValueError('functional observation split or digest differs')
            raw = torch.load(path, map_location='cpu', weights_only=True)
            op = raw['operator'].coalesce()
            rows, cols = op.indices().numpy()
            keep = observed.numpy()[cols]
            matrix = csr_matrix((op.values().numpy()[keep], (rows[keep], cols[keep])), shape=op.shape)
            mass = np.asarray(matrix.sum(1)).ravel()
            reliable = torch.from_numpy(mass >= .5)
            matrix = (diags(1 / np.maximum(mass, 1e-8)) @ matrix).tocsr()
            examples = []
            for region in range(len(raw['masks'])):
                episode = training_episode(raw, region, observed)
                if episode is None:
                    continue
                positive, negative, _ = episode
                pos = torch.where(positive & reliable)[0]
                neg = torch.where(negative & reliable)[0]
                if len(pos) and len(neg):
                    examples.append(dict(region=region, positive=pos, negative=neg))
            view = dict(frame=frame, operator=matrix, examples=examples,
                        semantic=raw['semantic_pixels'], local=raw['local_pixels'],
                        descriptors=raw['semantic_descriptors'])
            self.examples.extend((len(self.views), i) for i in range(len(examples)))
            self.views.append(view)
        if not self.examples:
            raise ValueError('no reliable functional query comparisons')

    def sample(self, index, generator, pairs=64, device='cuda'):
        vi, ri = self.examples[index]
        view = self.views[vi]; example = view['examples'][ri]
        pixels = torch.cat([ids[torch.randint(len(ids), (pairs,), generator=generator)]
                            for ids in [example['positive'], example['negative']]])
        matrix = view['operator'][pixels.numpy()].tocoo()
        unique, inverse = np.unique(matrix.col, return_inverse=True)
        operator = torch.sparse_coo_tensor(
            torch.tensor(np.stack([matrix.row, inverse]), dtype=torch.long, device=device),
            torch.tensor(matrix.data, device=device), (len(pixels), len(unique))).coalesce()
        queries = dict(semantic=view['descriptors'][example['region']], local=view['local'][pixels[0]])
        teacher, queries = ({}, {k: F.normalize(v.to(device).float(), dim=0) for k, v in queries.items()})
        for channel in ['semantic', 'local']:
            teacher[channel] = F.normalize(view[channel][pixels].to(device).float(), dim=-1) @ queries[channel]
        return torch.from_numpy(unique), operator, queries, teacher


def unchanged_state(field):
    from .relation_training import buffer_digests
    buffers = {k: v for k, v in buffer_digests(field).items() if k not in {'semantic.codes', 'local.codes'}}
    parameters = {k: p.detach().clone() for k, p in field.named_parameters()}
    return buffers, parameters


@torch.no_grad()
def source_audit(field, maps, source, device, seed=20261010):
    """Fixed held-out source preferences; no optimization or model selection."""
    generator = torch.Generator().manual_seed(seed)
    values = {k: dict(parent_kl=[], refined_kl=[]) for k in maps}
    for index in range(len(source.examples)):
        rows, operator, queries, teacher = source.sample(index, generator, device=device)
        for channel, adapter in maps.items():
            codes = getattr(field, channel).codes[rows].to(device).float()
            for name, decoded in [('parent', codes), ('refined', adapter(codes))]:
                score = projected_query_scores(decoded, queries[channel], operator)
                values[channel][name + '_kl'].append(float(preference_loss(score, teacher[channel])))
    return dict(regions=len(source.examples), frames=[v['frame'] for v in source.views], seed=seed,
                metrics={k: {n: float(np.mean(v)) for n, v in row.items()} for k, row in values.items()},
                used_for_optimization=False, used_for_selection=False)


def fit(parent, observations, output, *, epochs=2, seed=20261009, learning_rate=.0001,
        pairs=64, device='cuda', field_loader=None, save_function=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    loader = load_field if field_loader is None else field_loader
    writer = save_field if save_function is None else save_function
    field, manifest = loader(parent, 'cpu')
    source_digest = sha256(Path(observations) / 'observations.json')
    if manifest['provenance']['observations_sha256'] != source_digest:
        raise ValueError('functional source and parent construction differ')
    if any(getattr(field, c).weight is not None for c in ['semantic', 'local']):
        raise ValueError('this experiment requires the declared full-native parent')
    before, parameters = unchanged_state(field)
    source = FunctionalSource(observations, field.observed)
    if source.manifest['world_units_per_metre'] != field.world_units_per_metre:
        raise ValueError('source/field world units differ')
    maps = nn.ModuleDict({c: NativeResidual(getattr(field, c).native_dim) for c in ['semantic', 'local']}).to(device)
    optimizer = torch.optim.AdamW(maps.parameters(), lr=learning_rate, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed)
    visits = torch.zeros(len(source.examples), dtype=torch.long)
    history = []; steps = epochs * len(source.examples); step = 0
    for epoch in range(epochs):
        for index in torch.randperm(len(source.examples), generator=generator).tolist():
            rows, operator, queries, teacher = source.sample(index, generator, pairs, device)
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            for channel, adapter in maps.items():
                original = getattr(field, channel).codes[rows].to(device).float()
                decoded = adapter(original)
                scores = projected_query_scores(decoded, queries[channel], operator)
                functional = preference_loss(scores, teacher[channel])
                trust = (1 - (decoded * F.normalize(original, dim=-1)).sum(-1)).mean()
                loss = .5 * (functional + trust)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('nonfinite functional native objective')
                loss.backward()
                losses[channel] = dict(functional=float(functional.detach()), trust=float(trust.detach()))
            torch.nn.utils.clip_grad_norm_(maps.parameters(), 1.)
            optimizer.step(); visits[index] += 1; step += 1
            if step % 100 == 0 or step == steps:
                status = dict(step=step, steps=steps, losses=losses)
                history.append(status); print('functional_native', status, flush=True)
    if not bool((visits == epochs).all()):
        raise ValueError('incomplete functional region exposure')
    # Open held-out source observations only after the final update; no early
    # stopping, parameter search, or benchmark inputs enter this fitter.
    development = FunctionalSource(observations, field.observed, 'development')
    audit = source_audit(field, maps, development, device)
    drift = {}
    with torch.no_grad():
        for channel, adapter in maps.items():
            codes = getattr(field, channel).codes
            error, count = 0., 0
            for start in range(0, len(codes), 8192):
                stop = min(start + 8192, len(codes))
                eligible = field.observed[start:stop]
                original = codes[start:stop][eligible].to(device).float()
                decoded = adapter(original)
                if not bool(torch.isfinite(decoded).all()):
                    raise ValueError('nonfinite baked native coordinates')
                error += float((1 - (F.normalize(original, dim=-1) * decoded).sum(-1)).sum())
                count += len(decoded)
                codes[start:stop][eligible] = decoded.cpu().to(codes.dtype)
            drift[channel] = error / max(count, 1)
    after, _ = unchanged_state(field)
    if after != before or any(not torch.equal(p, dict(field.named_parameters())[k]) for k, p in parameters.items()):
        raise ValueError('functional fit modified protected field state')
    torch.save({k: v.cpu() for k, v in maps.state_dict().items()}, output / 'construction_maps.pt')
    exported = writer(field, output / 'field', dict(parent_field_sha256=manifest['files']['field.pt']['sha256'],
        observations_sha256=source_digest, supervision='source-only native pair preferences',
        construction='full-dimensional affine residuals baked into native Gaussian coordinates',
        benchmark_queries_used=False, additional_3d_labels=False))
    report = dict(parent=str(parent), parent_sha256=manifest['files']['field.pt']['sha256'],
        parent_source_schema=manifest['schema'],
        observations=str(observations), observations_sha256=source_digest, epochs=epochs, seed=seed,
        learning_rate=learning_rate, pairs=pairs, temperature=.1, trust_weight=1., weight_decay=.01,
        steps=steps, training_regions=len(source.examples), each_region_visits=epochs,
        source_frames=[v['frame'] for v in source.views], history=history, source_development=audit,
        mean_direction_drift=drift, protected_state_unchanged=True,
        construction_maps_sha256=sha256(output / 'construction_maps.pt'),
        export_sha256=field_identity(exported), field_schema=exported['schema'],
        export_state_identity=exported.get('state_identity'), deployment_bytes=deployment_bytes(exported),
        seconds=time.monotonic() - started,
        source_access_at_query=False, benchmark_queries_used=False)
    (output / 'training.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
