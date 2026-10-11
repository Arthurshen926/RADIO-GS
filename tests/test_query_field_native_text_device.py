from types import SimpleNamespace
import pytest
import torch
from radio_gs.query_field.field import NativeCoordinates
from radio_gs.query_field.semantic_measure import SemanticMeasure
from radio_gs.query_field.scope_confidence import native_text_evidence
from radio_gs.query_field.native_text_device import NativeTextDevice
from test_query_field_semantic_measure import measure


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_device_mirror_matches_full_native_execution_without_mutating_field(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(82)
    f=SimpleNamespace(xyz=torch.randn(12,3),observed=torch.ones(12,dtype=torch.bool),
        semantic=NativeCoordinates(torch.randn(12,9)),semantic_measure=measure(),
        evidence=SimpleNamespace(region_semantic=torch.randn(5,9)))
    before={k:v.clone() for k,v in f.semantic_measure.state_dict().items()}
    reader=NativeTextDevice(f,device)
    reference=SimpleNamespace(observed=f.observed.to(device),semantic=NativeCoordinates(f.semantic.codes.to(device)),
        semantic_measure=SemanticMeasure({k:v.to(device) for k,v in before.items()},12),
        evidence=SimpleNamespace(region_semantic=f.evidence.region_semantic.to(device)))
    negative=torch.randn(4,9)
    for q in torch.randn(3,9):
        actual=reader(q,negative);expected=native_text_evidence(reference,q.to(device),negative.to(device))
        for a,b in zip(actual,expected):assert torch.equal(a,b.cpu())
    for k,v in before.items():assert torch.equal(v,f.semantic_measure.state_dict()[k])
    assert f.semantic.codes.device.type=='cpu'
