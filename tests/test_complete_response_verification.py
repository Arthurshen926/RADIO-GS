import numpy as np
import pytest
from scipy import sparse
from scipy.stats import rankdata

from radio_gs.querying.complete_response_verification import (
    CompleteResponseVerifier,recover_second_teacher_ranks,
)
from radio_gs.querying.observation_readout import mean_native_identity_ranks


def test_recover_native_rank_fusion_with_ties_and_float32_rounding():
    rng=np.random.default_rng(3)
    native=rng.normal(size=6000).astype('float32')
    second=rng.integers(0,200,size=6000).astype('float32')
    available=np.arange(6000)[::2]
    fused=mean_native_identity_ranks(native,second,available)
    recovered,detail=recover_second_teacher_ranks(native,fused,available)
    np.testing.assert_array_equal(recovered[available],rankdata(second[available]))
    assert not detail['second_teacher_cosine_recovered']
    with pytest.raises(ValueError):recover_second_teacher_ranks(native,np.zeros(6000),available)


def test_query_verifier_prefers_correspondence_and_excludes_all_proposal_views():
    # Both instances visible in each frame; first frame is proposal-only.
    membership=sparse.csc_matrix([[1,0,1,0],[0,1,0,1]],dtype=float)
    reader=CompleteResponseVerifier(membership,sparse.csc_matrix(np.ones((2,2))),[0,0,1,1])
    masks=[np.array([1,0],bool),np.array([0,1],bool)]
    winner,detail=reader.verify(masks,[0,1],np.array([100,0,0,1]),np.array([100,0,0,1]))
    assert winner==1 and detail['common_views']==[1]
    assert detail['excluded_proposal_views']==[0]
    assert detail['scores']==pytest.approx([-.5,.5])
    _,control=reader.verify(masks,[0,1],np.array([100,0,0,1]),np.array([100,0,0,1]),permuted=True)
    assert control['common_views']==detail['common_views'] and control['query_budget_error']==0


def test_no_common_evidence_or_query_tie_preserves_original_response():
    membership=sparse.eye(2,format='csc')
    reader=CompleteResponseVerifier(membership,sparse.csc_matrix(np.ones((2,1))),[0,0])
    masks=[np.array([1,0],bool),np.array([0,1],bool)]
    winner,detail=reader.verify(masks,[0,1],np.zeros(2),np.zeros(2))
    assert winner==0 and detail['no_common_evidence']
