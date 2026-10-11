import numpy as np
from scripts.research.audit_query_field_extension_quality import compare_masks


def test_large_correct_core_does_not_certify_extension():
    gt=np.array([1]*90+[0]*10,dtype=bool)
    initial=gt.copy();expanded=np.ones(100,dtype=bool)
    r=compare_masks(initial,expanded,gt)
    assert r['expanded_precision']==.9 and r['added_precision']==0
    assert r['harmful'] and r['added_fp']==10


def test_marginal_threshold_and_nonmonotone_replay():
    gt=np.array([1,1,0,0],dtype=bool)
    r=compare_masks(np.array([1,0,0,0],dtype=bool),np.array([1,1,1,0],dtype=bool),gt)
    assert r['beneficial'] and r['added_precision']>r['break_even_added_precision']
    r=compare_masks(np.array([1,0,1,0],dtype=bool),np.array([0,1,0,0],dtype=bool),gt)
    assert r['removed_tp']==1 and r['removed_fp']==1 and r['added_tp']==1
