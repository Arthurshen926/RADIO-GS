from types import SimpleNamespace
import pytest
import torch
from scripts.research.run_tmm_support_agreement import PredictionMemory, source_guide


def test_transient_seal_rejects_changed_payload_and_bounds_memory(tmp_path):
    run=SimpleNamespace(root=tmp_path)
    memory=PredictionMemory(limit=8)
    report=memory.add(run,'scope',torch.tensor([[.2],[.3]]))
    assert report['bytes']==8
    assert torch.equal(memory.read(run,'scope',1),torch.tensor([[.2],[.3]]))
    with pytest.raises(ValueError,match='RAM budget'):
        memory.add(run,'matched',torch.ones(1,1))
    key=(str(tmp_path),'scope')
    value=memory.values[key].copy();value[0,0]=.9
    memory.values[key]=value
    with pytest.raises(ValueError,match='sealed transient'):
        memory.read(run,'scope',1)


def test_frozen_guide_requires_aligned_carrier_and_original_validity():
    data=dict(category_logits=torch.tensor([[2.,1.],[0.,1.]]),semantic_observed=torch.tensor([True,False]))
    assert torch.equal(source_guide(data,'scannet'),torch.tensor([[True,False],[False,False]]))
    data=dict(semantic_geometry={'sha256':'a'},selection_geometry={'sha256':'b'})
    with pytest.raises(ValueError,match='carriers differ'):
        source_guide(data,'lerf')
