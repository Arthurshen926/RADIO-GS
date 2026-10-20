"""Change only support lookup; independently replay from construction rasters."""
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import sparse
import torch

from .contracts import Support
from .element_relation_training import view_transfer
from .mixture_paired_evidence import KIND, MixturePairedEvidence
from .relation_training import SourceRelations
from .storage import load_field_versioned, sha256, field_identity, deployment_bytes
from .research_storage import research_writer,method_binding,tensor_state_identity,storage_binding


@torch.no_grad()
def compile_field(parent, observations, output, *, confidence_weighted=False, store_root=None, legacy_v1=False):
    writer=research_writer(store_root=store_root,legacy_v1=legacy_v1)
    output = Path(output)
    if output.exists():raise FileExistsError(output)
    started = time.monotonic(); field, manifest = load_field_versioned(parent,store_root=store_root)
    parent_binding=method_binding(parent,manifest)
    original_tensor_identity=tensor_state_identity(field)
    output.mkdir(parents=True, exist_ok=False)
    if field.evidence_kind != 'soft_partial_view_evidence_v1':
        raise ValueError('matched soft reference required')
    source_path = Path(observations) / 'observations.json'
    if manifest['provenance']['observations_sha256'] != sha256(source_path):
        raise ValueError('source geometry/rows differ')
    source = SourceRelations(observations, field.observed)
    if source.manifest['world_units_per_metre'] != field.world_units_per_metre:
        raise ValueError('physical units differ')
    if confidence_weighted:
        from .confidence_mixture_evidence import KIND as evidence_kind, ConfidenceMixtureEvidence as relation_class
    else:
        evidence_kind, relation_class = KIND, MixturePairedEvidence
    relation = relation_class(field.evidence.state_dict(), len(field.xyz))
    queries = []
    for view in source.views:
        pixels = view['pixels'][0][torch.tensor([0, len(view['pixels'][0]) // 2])]
        anchor = view['best'][pixels[:1]].long()
        queries.append((view['frame'], 'point', Support(anchor, torch.ones(1)), float(view['scales'][0])))
        # Actual full projection contributors, not a two-dominant-anchor proxy.
        indices, weights = [], []
        for pixel, weight in zip(pixels.tolist(), [.25, .75]):
            ray = view['columns'].getrow(pixel)
            keep = field.observed[ray.indices].numpy() & (ray.data > 0)
            ids, values = ray.indices[keep], ray.data[keep]
            if not len(ids): raise ValueError('source replay ray has no observed support')
            indices.extend(ids.tolist()); weights.extend((weight * values / values.sum()).tolist())
        queries.append((view['frame'], 'two_full_rays', Support(torch.tensor(indices), torch.tensor(weights)),
                        float(view['scales'][0])))
    target_rows = np.unique(np.r_[np.linspace(0, len(field.xyz) - 1, min(512, len(field.xyz)), dtype=np.int64),
                                 np.concatenate([q[2].indices.numpy() for q in queries])])
    support_rows = np.unique(np.concatenate([q[2].indices.numpy() for q in queries]))
    # Recreate only the source-side operands needed by the fixed replay. Runtime
    # never sees these masks/cameras. Combine masks BEFORE lifting target rows.
    sources = []
    for view in source.views:
        full, peak = view_transfer(view['columns'], field.observed, minimum_peak=0.)
        transfer = (sparse.diags(np.minimum(1., peak / .1)) @ full).tocsr()
        masks = [sparse.csr_matrix(view[k].numpy().astype(np.float32)) for k in ['positive', 'negative']]
        sources.append((view, [(transfer[support_rows] @ m.T).toarray() for m in masks],
                        transfer[target_rows], masks))
    replay = []
    for frame, kind, support, scale in queries:
        expected = [np.zeros(len(target_rows), np.float32), np.zeros(len(target_rows), np.float32)]
        ids = np.searchsorted(support_rows, support.indices.numpy())
        weights = support.weights.numpy() / float(support.weights.sum())
        expected_coefficients = []
        for view, (p, n), transfer, masks in sources:
            p, n = p[ids], n[ids]
            known = p + n; ratio = view['scales'].numpy() / scale
            admitted = (p > n) & (ratio >= .5)[None, :] & (ratio <= 2)[None, :]
            preference = np.where(admitted, p if confidence_weighted else p / np.maximum(known, 1e-12), 0)
            confidence = np.minimum(np.where(admitted, known, 0).max(1), 1.)
            association = preference / np.maximum(preference.sum(1, keepdims=True), 1e-12)
            coefficient = (association * (confidence * weights)[:, None]).sum(0)
            expected_coefficients.extend(coefficient.tolist())
            for k, mask in enumerate(masks): expected[k] += transfer @ (mask.T @ coefficient)
        actual = relation.masses(support, scale, torch.from_numpy(target_rows))
        errors = [float(np.max(np.abs(a.numpy() - b))) for a, b in zip(actual, expected)]
        coefficient = relation.coefficients(support, scale)
        coefficient_error = float(np.max(np.abs(coefficient.numpy() - expected_coefficients)))
        if max(*errors, coefficient_error) > 1e-5:
            raise ValueError('source/raster and exported mixture functions differ')
        replay.append(dict(source_frame=frame, kind=kind, support_rows=len(support.indices), scale=scale,
            positive_error=errors[0], negative_error=errors[1], coefficient_error=coefficient_error,
            known_elements=int(((actual[0] + actual[1]) > 0).sum()),
            admitted_views=int(torch.unique(relation.views[coefficient > 0]).numel())))
    # A function change keeps all tensor bytes but needs its own scientific spec.
    # Constructors may normalize dtypes; retain each original buffer exactly.
    for name,value in field.evidence.named_buffers():
        module=relation
        parts=name.split(".")
        for part in parts[:-1]:module=getattr(module,part)
        module._buffers[parts[-1]]=value
    field.evidence=relation;field.evidence_kind=evidence_kind
    if tensor_state_identity(field)!=original_tensor_identity:
        raise ValueError("mixture function changed original tensor state")
    provenance=copy.deepcopy(manifest['provenance'])
    provenance.update(support_lookup='mixture of confidence-weighted partial associations',
                      benchmark_queries_used=False,additional_3d_labels=False)
    destination=output/'field'
    if legacy_v1 and manifest["schema"].endswith(".v1"):
        destination.mkdir()
        shutil.copyfile(Path(parent)/'field.pt',destination/'field.pt')
        if sha256(destination/'field.pt')!=manifest['files']['field.pt']['sha256']:raise ValueError('copied payload differs')
        exported=copy.deepcopy(manifest);exported['spec']['evidence_kind']=evidence_kind;exported['provenance']=provenance
        (destination/'manifest.json').write_text(json.dumps(exported,indent=2)+'\n')
    else:
        exported=writer(field,destination,provenance)
    report = dict(parent_sha256=field_identity(manifest),
        parent_binding=parent_binding,export_binding=method_binding(destination,exported),
        storage=storage_binding(store_root=store_root,legacy_v1=legacy_v1),
        tensor_state_identity=original_tensor_identity,field_schema=exported['schema'],
        complete_field_bytes=deployment_bytes(exported),
        export_sha256=field_identity(exported), exact_tensor_payload=True,
        protected_state_unchanged=True, source_frames=[v['frame'] for v in source.views],
        source_regions=sum(len(v['scales']) for v in source.views), training_steps=0,
        field_tensor_bytes=exported['tensor_bytes'], additional_tensor_bytes=0,
        replay=replay, replay_target_rows=target_rows.tolist(),
        source_access_at_query=False, benchmark_queries_used=False,
        replay_scope='all valid source views; point and two full rays; fixed sampled target rows plus all support rows',
        complete_v10_function_preservation=False, regional_confidence_retained=confidence_weighted,
        seconds=time.monotonic() - started)
    (output / 'construction.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
