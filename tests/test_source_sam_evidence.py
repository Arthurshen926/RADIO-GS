import json
import pytest
import torch
from radio_gs.training.source_sam_evidence import read_manifest,load_source_sam
from radio_gs.utils.immutable_artifacts import sha256_file


def test_source_manifest_binds_query_free_payload_to_exact_rgb(tmp_path):
    image=tmp_path/'source.png';image.write_bytes(b'source fixture')
    contract=dict(source='official_sam3_interactive_grid_multimask_hierarchy',official_decoder=True,query_free=True,complex_rope_buffers_preserved=True)
    payload=dict(scores=torch.tensor([.9]),packed_masks=torch.zeros(1,2,1,dtype=torch.uint8),mask_shape=[2,3],
                 metadata=dict(contract,image=str(image),source_image_sha256=sha256_file(image)))
    blob=tmp_path/'mask.pt';torch.save(payload,blob)
    row=dict(image=str(image),source_image_sha256=sha256_file(image),output=str(blob),output_sha256=sha256_file(blob))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(generation_contract=contract,images=[row])))
    doc,records=read_manifest(manifest)
    actual,_=load_source_sam(doc,records,image,row['source_image_sha256'])
    assert torch.equal(actual['scores'],payload['scores'])
    image.write_bytes(b'changed RGB')
    with pytest.raises(ValueError,match='RGB identity'):load_source_sam(doc,records,image,row['source_image_sha256'])
    manifest.write_text(json.dumps(dict(generation_contract=dict(contract,query_free=False),images=[row])))
    with pytest.raises(ValueError,match='query-free'):read_manifest(manifest)


def test_multiscale_adapter_rejects_corrupt_manifest_and_legacy_rotary_contract(tmp_path):
    from radio_gs.models.sam3_multiscale_hierarchy import canonical_json_sha256
    document=dict(contract='official-sam3-query-free-multiscale-hierarchy-manifest-v1',
                  generation_contract=dict(query_free=True,complex_rope_buffers_preserved=True),images=[])
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(document,content_digest='0'*64)))
    with pytest.raises(ValueError,match='digest differs'):read_manifest(path)
    document['generation_contract']['complex_rope_buffers_preserved']=False
    path.write_text(json.dumps(dict(document,content_digest=canonical_json_sha256(document))))
    with pytest.raises(ValueError,match='complex-RoPE'):read_manifest(path)


def test_automatic_legacy_cache_requires_explicit_control_opt_in(tmp_path):
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(generation_contract=dict(query_free=True),images=[])))
    with pytest.raises(ValueError,match='complex-RoPE'):read_manifest(path)
    document,records=read_manifest(path,allow_legacy=True)
    assert records=={} and 'complex_rope_buffers_preserved' not in document['generation_contract']
