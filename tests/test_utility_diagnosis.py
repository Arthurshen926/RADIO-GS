import numpy as np
import pytest
from radio_gs.response_study.utility_diagnosis import range_audit,gain_audit


def test_domain_bounds_are_fitted_only_on_source_and_keep_constant_features():
    r=range_audit([[0,2],[1,2]],[[3,2],[.5,4]],['varying','constant'])
    assert r['features'][0]['fit_max']==1
    assert r['features'][1]['constant_in_fit']
    assert [v['outside_fit_count'] for v in r['features']]==[1,1]
    with pytest.raises(ValueError):range_audit([[0,np.nan]],[[1,2]],['a','b'])


def test_gain_diagnosis_keeps_scene_macro_and_calibrated_abstention():
    rows=[dict(scene=s,query_index=i,ious=u,gains=g,evidence_available=True) for i,(s,u,g) in enumerate([
        ('a',[.5,.1],[0,.8]),('a',[.4,.8],[0,.7]),('b',[.3,.6],[0,.6])])]
    fixed=gain_audit(rows,.75)
    assert fixed['harmed']==1 and fixed['changed']==1
    assert fixed['scene_macro']['actual_gain']==pytest.approx(-.1)
    assert fixed['scene_macro']['actual_gain']==pytest.approx(
        fixed['scene_macro']['available_gain']-fixed['scene_macro']['regret'])
    rejected=gain_audit(rows,None)
    assert rejected['changed']==0 and rejected['scene_macro']['actual_gain']==0
    assert rejected['largest_predicted_replacement']['top_actual_gain']==pytest.approx(-.4)
