import builtins
import numpy as np
import pytest
import torch
from sklearn.ensemble import ExtraTreesRegressor
from radio_gs.query_field.region_quality import RegionQuality, compile_region_quality, TARGETS
from radio_gs.querying.conditional_region_membership import FEATURE_NAMES


def fixture():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(200, 16)).astype(np.float32)
    y = rng.uniform(size=(200, 3))
    model = ExtraTreesRegressor(n_estimators=7, max_depth=5, random_state=13, n_jobs=1).fit(x, y)
    checkpoint = dict(estimator=model, target_order=list(TARGETS), feature_names=FEATURE_NAMES, addition_threshold=.8)
    return x, model, compile_region_quality(checkpoint)


def test_native_three_output_predictions_and_cold_no_files(monkeypatch, tmp_path):
    x, model, state = fixture()
    path = tmp_path/'quality.pt'; torch.save(state, path)
    compiled = RegionQuality(torch.load(path, weights_only=True))
    reference = model.predict(x)
    def denied(*a, **k):raise AssertionError('runtime file read')
    monkeypatch.setattr(builtins, 'open', denied)
    actual = compiled(torch.from_numpy(x), chunk_size=31).numpy()
    np.testing.assert_array_equal(actual, reference)
    assert compiled(torch.empty(0, 16)).shape == (0, 3)


def test_added_quality_cannot_hide_behind_shared_core():
    _, _, state = fixture(); f = RegionQuality(state)
    p = torch.tensor([[.99, .95, .1], [.81, .6, .9], [.81, .6, .1], [.99, .95, .9]], dtype=torch.float64)
    assert f.eligible(p, torch.tensor([1., 1., 0., 0.]), torch.tensor([False, False, False, True])).tolist() == [False, True, True, False]


def test_preserve_double_threshold_tie():
    _, _, state = fixture(); f = RegionQuality(state)
    t = f.addition_threshold
    p = torch.tensor([[1., 1., float(t)], [1., 1., np.nextafter(float(t), -np.inf)]], dtype=torch.float64)
    assert f.eligible(p, torch.ones(2), torch.zeros(2, dtype=torch.bool)).tolist() == [True, False]


def test_split_threshold_neighbors_match_reference():
    x, model, state = fixture()
    probes=[]
    for tree in model.estimators_:
        for feature, cut in zip(tree.tree_.feature, tree.tree_.threshold):
            if feature < 0:continue
            for value in [np.float32(cut), np.nextafter(np.float32(cut), np.float32(-np.inf)), np.nextafter(np.float32(cut), np.float32(np.inf))]:
                row=x[0].copy();row[feature]=value;probes.append(row)
    x=np.array(probes)
    np.testing.assert_array_equal(RegionQuality(state)(torch.from_numpy(x)).numpy(), model.predict(x))


def test_reject_nonfinite_input_and_invalid_topology():
    _, _, state = fixture(); f = RegionQuality(state)
    with pytest.raises(ValueError):f(torch.full((1, 16), float('nan')))
    broken = {k:v.clone() for k,v in state.items()};broken['left'][0]=0
    with pytest.raises(ValueError):RegionQuality(broken)
    broken = {k:v.clone() for k,v in state.items()};broken['depth']=torch.tensor(0)
    with pytest.raises(ValueError):RegionQuality(broken)(torch.zeros(1,16))


def test_per_view_policy_replaces_seed_and_keeps_stable_tie():
    from radio_gs.query_field.region_admission import admit_regions
    _, _, state = fixture(); f = RegionQuality(state)
    args=dict(seed=0,candidates=torch.tensor([1,2,3]),intersection=torch.tensor([1.,1.,1.]),
              area=torch.tensor([2.,3.,3.,3.]),frames=torch.tensor([0,0,1,1]),view_count=2)
    p=torch.tensor([[.9,.9,.9],[.8,.8,.9],[.8,.8,.9]],dtype=torch.float64)
    w,c,ids=admit_regions(f,p,**args)
    assert ids.tolist()==[1,2] and w[0]==0 and w[3]==0
    torch.testing.assert_close(c,torch.tensor([.9,.8]))
    p[:,2]=.1
    assert admit_regions(f,p,**args) is None
