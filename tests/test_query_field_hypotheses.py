import pytest
import torch

from radio_gs.query_field import QueryPacket,Support
from radio_gs.query_field.field import NativeCoordinates
from radio_gs.query_field.hypotheses import propose,query,reverse_local_match
from test_query_preserving_field import example


def test_default_hypothesis_replays_original_and_protection_holds():
    field=example();packet=QueryPacket('image',semantic=field.semantic.codes[2],local=field.local.codes[2:4])
    before={k:v.clone() for k,v in field.state_dict().items()}
    original=field.query(packet)
    alternatives=propose(field,packet)
    assert len(alternatives)==field.candidate_count*3
    assert alternatives[0].scale==field.default_scale
    assert alternatives[0].anchor==int(original.candidate_indices[0])
    torch.testing.assert_close(alternatives[0].membership,original.membership_scores,rtol=0,atol=0)
    for policy in ['anchor','native_bidirectional']:
        result=query(field,packet,policy=policy)
        assert result.diagnostics['shared_member_core'] and not result.diagnostics['source_access']
        assert not result.diagnostics['identity_scores_calibrated']
    for name,value in field.state_dict().items():assert torch.equal(before[name],value)


def test_explicit_scope_and_registered_support_are_not_replaced():
    field=example()
    image=QueryPacket('image',semantic=field.semantic.codes[0],scale=.07)
    assert len(propose(field,image))==field.candidate_count
    assert all(h.scale==.07 for h in propose(field,image))
    point=QueryPacket('point_3d',positive=Support(torch.tensor([0]),torch.ones(1)),exact_positive=True)
    torch.testing.assert_close(query(field,point).membership_scores,field.query(point).membership_scores,rtol=0,atol=0)
    with pytest.raises(ValueError,match='native query evidence'):
        query(field,point,policy='native_bidirectional')
    with pytest.raises(ValueError,match='category unions'):
        propose(field,QueryPacket('text',semantic=field.semantic.codes[0],task='union'))


def test_reverse_patch_matching_penalizes_missing_query_parts_and_obeys_affine_codec():
    local=NativeCoordinates(torch.eye(2))
    patches=torch.eye(2)
    assert reverse_local_match(local,patches,torch.tensor([0]))==pytest.approx(.5)
    assert reverse_local_match(local,patches,torch.tensor([0,1]))==pytest.approx(1.)
    assert reverse_local_match(local,patches[0],torch.tensor([0]))==pytest.approx(1.)
    torch.manual_seed(8)
    codec=NativeCoordinates(torch.randn(9,3),torch.randn(5,3),torch.randn(5))
    q=torch.randn(4,5);rows=torch.tensor([1,4,8])
    expected=float((torch.nn.functional.normalize(codec.decode(rows),dim=-1)@
                    torch.nn.functional.normalize(q,dim=-1).T).max(0).values.mean())
    assert reverse_local_match(codec,q,rows,chunk_size=1)==pytest.approx(expected,abs=1e-6)
