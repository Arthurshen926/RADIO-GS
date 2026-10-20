"""Small, content-bound provenance overlays for immutable compiled observations.

Restoring view/region identities does not recover independent geometric
visibility. Query-time attachment reads only the overlay and supplied field,
never the source RGB, masks, cameras or historical predictions.
"""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def field_binding(payload):
    """Bind descriptors AND the operator; column counts alone cannot bind order."""
    names = ['descriptors', 'crow', 'columns', 'weights', 'supported']
    names += [k for k in ('member_identity_descriptors', 'member_identity_coverage') if k in payload]
    return dict(shape=list(payload['shape']), geometry=payload.get('geometry'),
                tensors={name: tensor_digest(payload[name]) for name in names})


def check_region_row(row, frame):
    """Return the cache's original proposal indices, not newly numbered masks."""
    import torch
    if row.get('frame') != frame:
        raise ValueError('source cache frame identity differs')
    descriptor, indices = row['descriptors'], row['native_proposal_indices']
    if (descriptor.ndim != 2 or indices.ndim != 1 or len(indices) != len(descriptor)
            or indices.dtype not in (torch.int32, torch.int64)
            or bool((indices < 0).any()) or len(indices.unique()) != len(indices)):
        raise ValueError('invalid original region proposal identities')
    return indices.tolist()


def verify_native_block(payload, row, frame, offset):
    import torch
    indices = check_region_row(row, frame)
    expected = row['descriptors'].half().cpu()
    actual = payload['descriptors'][offset:offset + len(indices)].cpu()
    if actual.dtype != expected.dtype or not torch.equal(actual, expected):
        raise ValueError('source descriptor order/content differs from compiled field')
    return indices


def verify_replacement_lineage(parent, replacement, records, frames, offsets):
    """Verify transfer lineage; this does NOT replay the replacement encoder."""
    import torch
    if (list(parent['shape']) != list(replacement['shape'])
            or parent.get('geometry') != replacement.get('geometry')):
        raise ValueError('replacement carrier/region binding differs')
    for key in ('crow', 'columns', 'weights', 'supported'):
        if parent[key].dtype != replacement[key].dtype or not torch.equal(parent[key], replacement[key]):
            raise ValueError('replacement transfer differs: ' + key)
    expected = [(f, hi-lo) for f, lo, hi in zip(frames, offsets[:-1], offsets[1:])]
    if [(r['frame'], r['regions']) for r in records] != expected:
        raise ValueError('replacement build region order/count differs')
    if len(replacement['descriptors']) != offsets[-1]:
        raise ValueError('replacement descriptor count differs')


def attach_provenance(payload, field_path, reference, *, source_frames=None):
    """Return a shallow overlay; all original tensors remain shared/unmodified."""
    import torch
    from .store import region_layout
    path = Path(reference['path'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('region provenance file hash differs')
    metadata = json.loads(raw)
    if metadata.get('schema') != 'radiogs.compiled_region_provenance.v1':
        raise ValueError('unknown region provenance schema')
    if Path(metadata['field']['path']).resolve() != Path(field_path).resolve():
        raise ValueError('region provenance points to a different field')
    if metadata['field']['binding'] != field_binding(payload):
        raise ValueError('region provenance field content differs')
    frames, offsets = metadata['source_frames'], metadata['region_offsets']
    region_layout(offsets, frames, payload['shape'][1])
    if source_frames is not None and list(source_frames) != frames:
        raise ValueError('scene source view order differs from region provenance')
    rows = metadata['regions_by_view']
    if len(rows) != len(frames):
        raise ValueError('region provenance view records differ')
    for frame, lo, hi, row in zip(frames, offsets[:-1], offsets[1:], rows):
        indices = row['proposal_indices']
        if (row['frame'] != frame or len(indices) != hi-lo or len(set(indices)) != len(indices)
                or any(type(x) is not int or x < 0 for x in indices)):
            raise ValueError('region provenance proposal identities differ')
    if ('region_offsets' in payload) != ('source_frames' in payload):
        raise ValueError('partial inline region provenance')
    if 'source_frames' in payload:
        if list(payload['source_frames']) != frames or payload['region_offsets'].tolist() != offsets:
            raise ValueError('inline and sidecar region provenance disagree')
    return dict(payload, source_frames=frames, region_offsets=torch.tensor(offsets, dtype=torch.int64),
                region_provenance=dict(path=str(path), sha256=reference['sha256'],
                    verification=metadata['verification'], visibility_recovered=False))
