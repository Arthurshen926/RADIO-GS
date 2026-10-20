import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from radio_gs.querying.centered_response_value import CenteredResponseValue
from radio_gs.querying.response_value import set_features,relative_utilities


def test_centered_value_is_invariant_to_query_shared_offsets_and_candidate_order():
    rng=np.random.default_rng(7)
    pools=[rng.integers(-5,6,size=(4,39)).astype(np.float32) for _ in range(12)]
    x=np.concatenate([set_features(p) for p in pools]);y=np.concatenate([p[:,0]-p[:,0].mean() for p in pools])
    model=CenteredResponseValue(ExtraTreesRegressor(n_estimators=8,random_state=3)).fit(x,y)
    pool=pools[0];offset=rng.integers(-10,11,size=39).astype(np.float32)
    np.testing.assert_array_equal(relative_utilities(model,pool),relative_utilities(model,pool+offset))
    order=[0,3,1,2]
    np.testing.assert_array_equal(relative_utilities(model,pool)[order],relative_utilities(model,pool[order]))
    assert model.estimator.n_features_in_==39 and not model.feature_importances_[:78].any()
