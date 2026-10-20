import numpy as np
import pytest
from radio_gs.querying.response_value import set_features, relative_utilities
from radio_gs.querying.residual_response_value import (
    structural_columns, pool_arrays, room_folds, out_of_room_prior, fit_residual_models,
)


def reports():
    rng=np.random.default_rng(3); result=[]
    for i in range(6):
        rows=[]
        for _ in range(4):
            x=rng.random((4,39),dtype=np.float32); null=x.copy();null[:,9:]=null[[2,3,0,1],9:]
            y=.2*x[:,0]+.6*x[:,9]
            rows.append(dict(features=x.tolist(),permuted_features=null.tolist(),ious=y.tolist()))
        result.append(dict(scene=f'room{i}',rows=rows))
    return result


PARAMS=dict(n_estimators=8,min_samples_leaf=1,max_leaf_nodes=32,random_state=4)


def test_structural_prior_cannot_read_query_coordinates():
    x=np.random.default_rng(0).random((4,39),dtype=np.float32); changed=x.copy();changed[:,9:]+=2
    np.testing.assert_allclose(structural_columns(set_features(x)),set_features(x,query_blind=True))
    np.testing.assert_array_equal(structural_columns(set_features(x)),structural_columns(set_features(changed)))


def test_out_of_room_targets_cannot_leak_into_their_prior():
    x,_,y,w,rooms=pool_arrays(reports()); assignment=room_folds(rooms,3)
    first,receipts=out_of_room_prior(x,y,w,rooms,PARAMS,3)
    held=np.array([assignment[s]==0 for s in rooms]); changed=y.copy();changed[held]+=100
    second,_=out_of_room_prior(x,changed,w,rooms,PARAMS,3)
    # Parallel tree predictions can add the same values in different orders.
    # Allow only float64 summation noise, not a target-dependent prediction.
    np.testing.assert_allclose(first[held],second[held],rtol=0,atol=1e-15)
    assert all(not set(r['fit_rooms']) & set(r['held_rooms']) for r in receipts)


def test_residual_model_is_candidate_equivariant_and_preserves_weight_mass():
    source=reports(); models,detail=fit_residual_models(source,PARAMS,3)
    x=np.array(source[0]['rows'][0]['features']); order=[0,3,1,2]
    for name in ('full','unregularized'):
        np.testing.assert_allclose(relative_utilities(models[name],x)[order],relative_utilities(models[name],x[order]))
    assert detail['augmented_weight']==pytest.approx(detail['original_weight'])
    assert detail['full_trees']==detail['no_null_control_trees']==16
    assert detail['null_zero_correction_mse'] < detail['unregularized_null_correction_mse']


def test_null_construction_rejects_changed_candidate_structure():
    data=reports(); data[0]['rows'][0]['permuted_features'][0][0]+=1
    with pytest.raises(ValueError,match='preserve candidates'):pool_arrays(data)
