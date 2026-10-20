import pytest
from radio_gs.querying.reference_admission import reference_choices


def records(p=1., n=0., a=.9, b=.8):
    return dict(semantic_memory=dict(positive_coverage=p, negative_violation=n, predicted_quality=a),
                constrained_memory=dict(predicted_quality=b))


def test_consistent_reference_is_not_reprompted_and_actual_conflict_is_repaired():
    assert reference_choices(records()) == dict(feasible='semantic_memory', quality='semantic_memory')
    assert reference_choices(records(n=.01))['feasible'] == 'constrained_memory'
    assert reference_choices(records(p=.99))['feasible'] == 'constrained_memory'


def test_quality_is_independent_and_ties_are_deterministic():
    assert reference_choices(records(b=.95)) == dict(feasible='semantic_memory', quality='constrained_memory')
    assert reference_choices(records(a=.9,b=.9))['quality'] == 'semantic_memory'
    with pytest.raises(ValueError): reference_choices(records(n=-.1))
