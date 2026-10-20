import numpy as np
import pytest
from radio_gs.querying.multitask_response_value import category_report, protected_output_rows
from radio_gs.querying.response_value import training_values
from radio_gs.querying.response_replacement import calibrate_margin, choose_response


def test_menu_calibration_protects_every_subset_with_one_candidate():
    # Candidate 1 helps the 19-class mean but harms the 10-class mean. A
    # menu-19-only guard would accept it; the shared output guard rejects it.
    menu = dict(scene='room', modality='category', gains=[0.,.2], evidence_available=True,
        ious=[.4,.5], menu_utilities={'19':[.4,.6], '15':[.4,.6], '10':[.4,.3]})
    expanded = protected_output_rows([menu])
    assert all(r['gains'] is menu['gains'] for r in expanded)
    selected, _ = calibrate_margin(expanded)
    assert choose_response(menu['gains'], selected['margin']) == 0
    assert selected['delta'] == 0


def test_room_task_balancing_and_no_development_fold_selection():
    x = np.zeros((3,39)).tolist()
    rows = [dict(fold=f,features=x,menu_utilities={'19':[.2,.4,.3],'15':[.3,.5,.4],'10':[.4,.6,.5]})
            for f in ('full',0,1,2)]
    report = dict(scene='s',rows=rows)
    fit = category_report(report,fit=True); dev = category_report(report,fit=False)
    assert len(fit['rows']) == 4 and [r['fold'] for r in dev['rows']] == ['full']
    np.testing.assert_allclose(dev['rows'][0]['ious'],[.3,.5,.4])
    instance = dict(rows=[dict(features=x,ious=[.1,.2,.3])]*17)
    _,_,weights=training_values([fit,instance])
    assert np.isclose(weights[:12].sum(),weights[12:].sum())
    rows[0]['menu_utilities'].pop('10')
    with pytest.raises(ValueError):
        category_report(report,fit=False)
