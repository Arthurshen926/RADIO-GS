"""Confidence-weighted target relation compilation; fixed hard associations."""
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch

from .compile_paired_evidence import csr_state
from .contracts import Support
from .element_relation_training import view_transfer
from .soft_paired_evidence import enable
from .relation_training import SourceRelations, buffer_digests
from .storage import load_field, save_field, sha256


@torch.no_grad()
def compile_field(parent, observations, output, hard_reference):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); field, manifest = load_field(parent)
    source_path = Path(observations) / 'observations.json'
    if manifest['provenance']['observations_sha256'] != sha256(source_path):
        raise ValueError('source geometry/rows differ from parent')
    protected = buffer_digests(field); parameters = {k: v.clone() for k, v in field.named_parameters()}
    source = SourceRelations(observations, field.observed)
    if source.manifest['world_units_per_metre'] != field.world_units_per_metre:
        raise ValueError('physical coordinate units differ')
    blocks = {k: [] for k in ['association', 'positive', 'negative']}
    transfers, views, scales, source_records = [], [], [], []
    for vi, view in enumerate(source.views):
        full, peak = view_transfer(view['columns'], field.observed, minimum_peak=0.)
        hard, _ = view_transfer(view['columns'], field.observed)
        soft = (sparse.diags(np.minimum(1., peak / .1)) @ full).tocsr()
        pos = sparse.csr_matrix(view['positive'].numpy().astype(np.float32)).T
        neg = sparse.csr_matrix(view['negative'].numpy().astype(np.float32)).T
        for name, matrix in [('association', hard @ pos), ('positive', soft @ pos), ('negative', soft @ neg)]:
            matrix = matrix.tocsr(); matrix.data = np.clip(matrix.data, 0, 1); matrix.eliminate_zeros()
            blocks[name].append(matrix)
        transfers.append((hard, soft)); views.extend([vi] * pos.shape[1]); scales.extend(view['scales'].tolist())
        source_records.append(dict(frame=view['frame'], regions=view['original_regions'],
            **{k + '_nnz': blocks[k][-1].nnz for k in blocks}))
        print('compiled confidence-weighted coordinates', view['frame'], {k: blocks[k][-1].nnz for k in blocks}, flush=True)
    state = dict(views=torch.tensor(views), scales=torch.tensor(scales))
    for prefix, values in blocks.items(): state.update(csr_state(sparse.hstack(values, format='csr'), prefix))
    del blocks
    enable(field, state)
    replay = []
    for view in source.views:
        pixels = view['pixels'][0]; ids = view['best'][pixels[torch.tensor([0, len(pixels) // 2])]].long()
        for multi in [False, True]:
            support = Support(ids if multi else ids[:1], torch.tensor([.25, .75]) if multi else torch.ones(1))
            scale = float(view['scales'][0]); expected = [np.zeros(len(field.xyz), np.float32) for _ in range(2)]
            for other, (hard, soft) in zip(source.views, transfers):
                support_pixels = np.asarray(hard[support.indices.numpy()].T @ support.weights.numpy()).ravel()
                association = other['positive'].numpy().astype(np.float32) @ support_pixels
                ratio = other['scales'].numpy() / scale
                association[(association < .5) | (ratio < .5) | (ratio > 2)] = 0
                association /= max(float(association.sum()), 1e-12)
                expected[0] += soft @ (other['positive'].numpy().T @ association)
                expected[1] += soft @ (other['negative'].numpy().T @ association)
            actual = field.evidence.masses(support, scale)
            errors = [float(np.max(np.abs(a.numpy() - e))) for a, e in zip(actual, expected)]
            if max(errors) > 1e-5: raise ValueError('soft relation function differs from source replay')
            replay.append(dict(source_frame=view['frame'], support=support.indices.tolist(),
                weights=support.weights.tolist(), scale=scale, positive_error=errors[0], negative_error=errors[1],
                known_elements=int(((actual[0] + actual[1]) > 0).sum())))
    # Validate every association entry against the completed hard reference.
    original = torch.load(Path(hard_reference) / 'field.pt', map_location='cpu', weights_only=True)
    for suffix in ['pointer', 'index', 'value']:
        if not torch.equal(state['association_' + suffix], original['evidence.positive_' + suffix]):
            raise ValueError('soft extension changed the fixed support association function')
    observed = field.observed; available = (state['positive_pointer'].diff() > 0) | (state['negative_pointer'].diff() > 0)
    old_available = (original['evidence.positive_pointer'].diff() > 0) | (original['evidence.negative_pointer'].diff() > 0)
    coverage = dict(observed_rows=int(observed.sum()), available_rows=int((observed & available).sum()),
        hard_available_rows=int((observed & old_available).sum()))
    del original
    after = buffer_digests(field)
    if any(after[k] != v for k, v in protected.items()): raise ValueError('compiler changed parent buffers')
    if any(not torch.equal(parameters[k], v) for k, v in field.named_parameters()): raise ValueError('compiler changed parent parameters')
    exported = save_field(field, output / 'field', dict(parent_field_sha256=manifest['files']['field.pt']['sha256'],
        observations_sha256=sha256(source_path), supervision='source-only confidence-weighted partial relation functions',
        additional_3d_labels=False, benchmark_queries_used=False, uncompressed_functional_reference=True))
    report = dict(parent_sha256=manifest['files']['field.pt']['sha256'], export_sha256=exported['files']['field.pt']['sha256'],
        source_frames=[v['frame'] for v in source.views], source_regions=sum(len(v['scales']) for v in source.views),
        source_records=source_records, replay=replay, protected_state_unchanged=True, training_steps=0,
        field_tensor_bytes=exported['tensor_bytes'], evidence_bytes=exported['tensor_bytes'] - manifest['tensor_bytes'],
        exact_hard_association_coordinates=True, relation_availability=coverage,
        hard_reference=dict(path=str(Path(hard_reference) / 'manifest.json'), sha256=sha256(Path(hard_reference) / 'manifest.json')),
        source_access_at_query=False, preserves_partial_relation_function=True,
        complete_v10_function_preservation=False, seconds=time.monotonic() - started)
    (output / 'construction.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
