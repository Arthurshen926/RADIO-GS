"""Compile and independently replay partial relation functions before queries."""
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch

from .contracts import Support
from .element_relation_training import view_transfer
from .paired_evidence import enable
from .relation_training import SourceRelations, buffer_digests
from .storage import load_field, save_field, sha256


def csr_state(matrix, prefix):
    matrix = matrix.tocsr(); matrix.eliminate_zeros(); matrix.sort_indices()
    return {prefix + '_pointer': torch.from_numpy(matrix.indptr.astype(np.int64)),
            prefix + '_index': torch.from_numpy(matrix.indices.astype(np.int64)),
            prefix + '_value': torch.from_numpy(matrix.data.astype(np.float32))}


@torch.no_grad()
def compile_field(parent, observations, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); field, manifest = load_field(parent)
    source_path = Path(observations) / 'observations.json'
    if manifest['provenance']['observations_sha256'] != sha256(source_path):
        raise ValueError('source geometry/rows differ from parent')
    protected = buffer_digests(field)
    parameters = {k: v.clone() for k, v in field.named_parameters()}
    source = SourceRelations(observations, field.observed)
    if source.manifest['world_units_per_metre'] != field.world_units_per_metre:
        raise ValueError('physical coordinate units differ')
    positives, negatives, transfers, views, scales, source_records = [], [], [], [], [], []
    for vi, view in enumerate(source.views):
        transfer, _ = view_transfer(view['columns'], field.observed)
        positive = (transfer @ sparse.csr_matrix(view['positive'].numpy().astype(np.float32)).T).tocsr()
        negative = (transfer @ sparse.csr_matrix(view['negative'].numpy().astype(np.float32)).T).tocsr()
        # Roundoff at 1 is not extra evidence. Unknown entries remain absent.
        positive.data = np.clip(positive.data, 0, 1)
        negative.data = np.clip(negative.data, 0, 1)
        positives.append(positive); negatives.append(negative)
        transfers.append(transfer)
        views.extend([vi] * positive.shape[1]); scales.extend(view['scales'].tolist())
        source_records.append(dict(frame=view['frame'], regions=view['original_regions'],
                                   positive_nnz=positive.nnz, negative_nnz=negative.nnz))
        print('compiled relation coordinates', view['frame'], positive.shape[1], positive.nnz, negative.nnz, flush=True)
    state = dict(views=torch.tensor(views), scales=torch.tensor(scales))
    for prefix, blocks in [('positive', positives), ('negative', negatives)]:
        state.update(csr_state(sparse.hstack(blocks, format='csr'), prefix))
    enable(field, state)
    # Source-derived, query-independent supports: one median reliable anchor in
    # each valid view plus a two-anchor uncertain support. Replay every Gaussian
    # against the pre-export per-view matrices, without benchmark inputs/labels.
    replay = []
    for vi, view in enumerate(source.views):
        pixels = view['pixels'][0]
        ids = view['best'][pixels[torch.tensor([0, len(pixels) // 2])]].long()
        for multi in [False, True]:
            support = Support(ids if multi else ids[:1], torch.tensor([.25, .75]) if multi else torch.ones(1))
            scale = float(view['scales'][0]); expected = [np.zeros(len(field.xyz), np.float32) for _ in range(2)]
            for other, transfer in zip(source.views, transfers):
                # Direct source route: project support to pixels, associate
                # masks, combine masks in raster space, then lift to Gaussians.
                # Runtime instead contracts only the exported Gaussian codes.
                support_pixels = np.asarray(transfer[support.indices.numpy()].T @ support.weights.numpy()).ravel()
                association = other['positive'].numpy().astype(np.float32) @ support_pixels
                ratio = other['scales'].numpy() / scale
                association[(association < .5) | (ratio < .5) | (ratio > 2)] = 0
                association /= max(float(association.sum()), 1e-12)
                expected[0] += transfer @ (other['positive'].numpy().T @ association)
                expected[1] += transfer @ (other['negative'].numpy().T @ association)
            actual = field.evidence.masses(support, scale)
            errors = [float(np.max(np.abs(a.numpy() - e))) for a, e in zip(actual, expected)]
            if max(errors) > 1e-5:
                raise ValueError('compiled partial-relation function differs from direct replay')
            replay.append(dict(source_frame=view['frame'], support=support.indices.tolist(),
                weights=support.weights.tolist(), scale=scale, positive_error=errors[0], negative_error=errors[1],
                known_elements=int(((actual[0] + actual[1]) > 0).sum())))
    after = buffer_digests(field)
    if any(after[k] != v for k, v in protected.items()):
        raise ValueError('compiler changed protected parent buffers')
    if any(not torch.equal(parameters[k], v) for k, v in field.named_parameters()):
        raise ValueError('compiler changed parent parameters')
    exported = save_field(field, output / 'field', dict(parent_field_sha256=manifest['files']['field.pt']['sha256'],
        observations_sha256=sha256(source_path), supervision='same source-only partial relation functions',
        additional_3d_labels=False, benchmark_queries_used=False, uncompressed_functional_reference=True))
    report = dict(parent_sha256=manifest['files']['field.pt']['sha256'], export_sha256=exported['files']['field.pt']['sha256'],
        source_frames=[v['frame'] for v in source.views], source_regions=sum(len(v['scales']) for v in source.views),
        source_records=source_records, replay=replay, protected_state_unchanged=True, training_steps=0,
        field_tensor_bytes=exported['tensor_bytes'], evidence_bytes=exported['tensor_bytes'] - manifest['tensor_bytes'],
        source_access_at_query=False, preserves_partial_relation_function=True,
        complete_v10_function_preservation=False, seconds=time.monotonic() - started)
    (output / 'construction.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
