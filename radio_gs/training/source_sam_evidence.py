"""Read query-free SAM evidence bound to an explicit source image manifest."""
import json
from pathlib import Path
import torch
from radio_gs.scripts.build_sam3_automatic_mask_cache import validate_automatic_mask_source_binding
from radio_gs.utils.immutable_artifacts import sha256_file


def read_manifest(path, *, allow_legacy=False):
    path=Path(path)
    document=json.loads(path.read_text())
    if document.get('generation_contract',{}).get('query_free') is not True:
        raise ValueError('source teacher requires query-free SAM proposals')
    if not allow_legacy and document['generation_contract'].get('complex_rope_buffers_preserved') is not True:
        raise ValueError('corrected complex-RoPE source evidence required; legacy controls require explicit opt-in')
    multiscale=document.get('contract')=='official-sam3-query-free-multiscale-hierarchy-manifest-v1'
    if multiscale:
        from radio_gs.models.sam3_multiscale_hierarchy import canonical_json_sha256
        from radio_gs.scripts.build_sam3_multiscale_hierarchy_cache import _load_source_authority
        unsigned={k:v for k,v in document.items() if k!='content_digest'}
        if canonical_json_sha256(unsigned)!=document['content_digest']:raise ValueError('multiscale manifest digest differs')
        authority=Path(document['source_authority']['path'])
        _,items=_load_source_authority(authority,document['source_authority']['sha256'])
        by_id={r['image_id']:r for r in items}
    rows={}
    for record in document['images']:
        if multiscale:
            source=by_id[record['image_id']];image=Path(source['path'])
            image=image if image.is_absolute() else authority.parent/image
            record=dict(record,image=str(image.resolve()),source_image_sha256=source['sha256'])
        key=str(Path(record['image']).resolve())
        if key in rows:raise ValueError('duplicate SAM source image')
        rows[key]=record
    return document,rows


def load_source_sam(document, records, image_path, image_sha256):
    key=str(Path(image_path).resolve())
    if key not in records:raise ValueError('source image absent from SAM manifest')
    record=records[key]
    if record['source_image_sha256']!=image_sha256 or sha256_file(image_path)!=image_sha256:
        raise ValueError('SAM source RGB identity differs')
    path=Path(record['output'])
    if sha256_file(path)!=record['output_sha256']:raise ValueError('SAM payload identity differs')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if document.get('contract')=='official-sam3-query-free-multiscale-hierarchy-manifest-v1':
        from radio_gs.models.sam3_multiscale_hierarchy import validate_multiscale_cache_payload
        expected={k:document[k] for k in ['generation_contract','source_authority','checkpoint','runtime_binding','producer_binding']}
        expected['source_image']=dict(image_id=record['image_id'],path=key,sha256=image_sha256,rgb_role='registered_source_or_mapping_view')
        validate_multiscale_cache_payload(payload,expected_metadata=expected)
        payload=dict(payload,scores=payload['quality'])
    else:
        validate_automatic_mask_source_binding(payload,Path(image_path),
            expected_generation_contract=document['generation_contract'],source_image_sha256=image_sha256)
    if len(payload['scores'])!=len(payload['packed_masks']) or not torch.isfinite(payload['scores']).all():
        raise ValueError('invalid SAM quality axes')
    return payload,record['output_sha256']
