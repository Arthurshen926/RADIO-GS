import pytest
from radio_gs.training.scope_family_split import scope_family_split


def test_parent_part_synonym_transitivity_never_crosses_folds():
    d=scope_family_split(range(8),[(0,1),(1,2),(3,4)],folds=3)
    assert d['usable']
    assert d['query_to_fold'][0]==d['query_to_fold'][2]
    assert sorted(sum(d['folds'],[]))==list(range(8))
    assert d==scope_family_split(reversed(range(8)),[(3,4),(1,2),(0,1)],folds=3)


def test_no_fabricated_holdout_when_all_scopes_connected():
    d=scope_family_split(range(3),[(0,1),(1,2)])
    assert not d['usable'] and max(map(len,d['folds']))==3
    with pytest.raises(ValueError):scope_family_split([0,1],[(0,2)])


def test_giant_family_is_not_made_valid_by_tiny_singletons():
    d=scope_family_split(range(10),((i,i+1) for i in range(6)),folds=3)
    assert d['enough_families'] and not d['usable']
    assert d['dominated_by_one_fold']
    assert len({d['query_to_fold'][i] for i in range(7)})==1
