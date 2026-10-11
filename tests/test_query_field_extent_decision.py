import itertools
import numpy as np
import torch
from radio_gs.query_field.extent_factors import ExtentFactors
from radio_gs.query_field.extent_decision import ExtentDecision, render_row, probability, CUTOFF


def fixture():
    # Redundant foreground, missing foreground, and two background footprints.
    state = dict(pointer=torch.tensor([0, 2, 3, 5]), index=torch.tensor([0, 1, 2, 0, 3]),
                 alpha=torch.tensor([.9, .8, .9, .7, .9]), atom=torch.zeros(3, dtype=torch.long))
    state['visibility'] = state['alpha'].clone()
    return ExtentFactors({'views.0.'+k: v for k, v in state.items()}, 4)


def reference(factors, selected, target, positive, negative, confidence, cost):
    member = torch.tensor(selected, dtype=torch.float64); result = .1 * np.dot(cost, selected)
    offset = 0
    for vi, f in enumerate(factors.views):
        if confidence[vi] == 0:
            offset += len(f.atom); continue
        p = torch.sigmoid(torch.logit(f(member).clamp(1e-6, 1-1e-6)) - torch.logit(torch.tensor(CUTOFF, dtype=torch.float64)))
        stop = offset + len(p)
        y = torch.tensor(target[offset:stop], dtype=torch.float64)
        pos = torch.tensor(positive[offset:stop]); neg = torch.tensor(negative[offset:stop]); known = pos | neg
        bce = torch.nn.functional.binary_cross_entropy(p.clamp(1e-5, 1-1e-5), y, reduction='none')
        loss = .5 * ((bce * pos).sum()/max(pos.sum(), 1) + (bce * neg).sum()/max(neg.sum(), 1))
        loss += 1 - (2*(p*y*known).sum()+1)/((p*known).sum()+(y*known).sum()+1)
        result += confidence[vi] / np.count_nonzero(confidence) * float(loss)
        offset = stop
    return result


def test_joint_objective_and_bidirectional_decisions():
    f = fixture(); decoder = ExtentDecision(f, 4)
    y = np.array([1., 1., 0.]); p = y.astype(bool); n = ~p
    cost = np.array([.3, .2, .2, .3]); conf = np.array([.8]); initial = np.array([1, 1, 0, 1], bool)
    selected, report = decoder.solve(initial, y, p, n, conf, cost)
    assert report['added'] > 0 and report['removed'] > 0
    assert np.all(np.diff(report['energy_trace']) <= 1e-9)
    assert abs(report['energy_trace'][0] - reference(f, initial, y, p, n, conf, cost)) < 2e-6
    assert abs(report['energy_trace'][-1] - reference(f, selected, y, p, n, conf, cost)) < 2e-6
    energies = [reference(f, np.array(s), y, p, n, conf, cost) for s in itertools.product([False, True], repeat=4)]
    assert abs(report['energy_trace'][-1] - min(energies)) < 2e-6


def test_exact_occlusion_cutoff_and_flip():
    ptr = np.array([0, 3]); ids = np.array([0, 1, 2]); alpha = np.array([.92, .999, .2], np.float32)
    f = ExtentFactors({'views.0.'+k: v for k, v in dict(pointer=torch.tensor(ptr), index=torch.tensor(ids), alpha=torch.tensor(alpha), visibility=torch.tensor(alpha), atom=torch.tensor([0])).items()}, 3)
    for bits in itertools.product([False, True], repeat=3):
        selected = np.array(bits)
        for flip in [-1, 0, 1, 2]:
            copy = selected.copy()
            if flip >= 0: copy[flip] = not copy[flip]
            assert abs(render_row(ptr, ids, alpha, selected, 0, flip) - float(f.views[0](torch.tensor(copy).float())[0])) < 1e-6
    # The hit crossing T < 1e-4 must be omitted, not evaluated as 1-prod(1-alpha).
    assert abs(render_row(ptr, ids, alpha, np.ones(3, bool), 0) - .92) < 1e-6


def test_empty_source_keeps_native_and_invalid_evidence_rejected():
    import pytest
    decoder = ExtentDecision(fixture(), 4); initial = np.array([1, 0, 0, 1], bool)
    y = np.array([1., 1., 0.]); p = y.astype(bool); n = ~p
    selected, report = decoder.solve(initial, y, p, n, np.zeros(1), np.ones(4))
    assert np.array_equal(selected, initial) and report['fallback']
    with pytest.raises(ValueError): decoder.solve(initial, y, p, p, np.ones(1), np.ones(4))


def test_float32_termination_matches_native_not_float64_product():
    # Native float32 stays just ABOVE 1e-4; float64 is just below it. Omitting
    # this last hit changes alpha by ~.08755, despite a tiny product difference.
    alpha = np.array([.7812169790267944, .5993716716766357, .9988591074943542], np.float32)
    assert np.prod(1-alpha.astype(np.float64)) < 1e-4
    t = np.float32(1.)
    for a in alpha: t = np.float32(t*np.float32(np.float32(1)-a))
    assert t > 1e-4
    actual = render_row(np.array([0, 3]), np.arange(3), alpha, np.ones(3, bool), 0)
    assert actual == np.float32(1)-t


def test_view_balanced_objective_with_soft_targets_and_unknown_rows():
    base = fixture().state_dict()
    state = dict(base, **{k.replace('views.0.', 'views.1.'): v.clone() for k, v in base.items()})
    factors = ExtentFactors(state, 4); decoder = ExtentDecision(factors, 4)
    target = np.array([.9, .1, .05, .95, .9, .7])
    positive = np.array([1, 0, 0, 1, 1, 0], bool)
    negative = np.array([0, 1, 1, 0, 0, 0], bool)
    initial = np.ones(4, bool); cost = np.array([.3, .2, .2, .3]); confidence = np.array([.2, .9])
    selected, report = decoder.solve(initial, target, positive, negative, confidence, cost)
    assert abs(report['energy_trace'][0] - reference(factors, initial, target, positive, negative, confidence, cost)) < 1e-7
    assert abs(report['energy_trace'][-1] - reference(factors, selected, target, positive, negative, confidence, cost)) < 1e-7
    # Unknown evidence cannot alter the decision by changing its target value.
    modified = target.copy(); modified[-1] = 0.
    second, second_report = decoder.solve(initial, modified, positive, negative, confidence, cost)
    assert np.array_equal(selected, second) and report['energy_trace'] == second_report['energy_trace']


def test_extent_field_roundtrip_preserves_parent(tmp_path):
    from test_query_preserving_field import example
    from radio_gs.query_field.storage import load_field, save_field
    field = example(); old = {k: v.clone() for k, v in field.state_dict().items()}
    field.extent_factors = ExtentFactors(fixture().state_dict(), len(field.xyz), field.opacities)
    save_field(field, tmp_path/'field', {})
    restored, manifest = load_field(tmp_path/'field')
    for k, v in old.items(): assert torch.equal(v, restored.state_dict()[k])
    assert manifest['external_scene_dependencies'] == []
    member = torch.ones(len(field.xyz))
    assert torch.equal(restored.extent_factors.views[0](member), field.extent_factors.views[0](member))
