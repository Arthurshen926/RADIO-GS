import itertools
import numpy as np
import torch
from radio_gs.query_field.extent_factors import ExtentFactors
from radio_gs.query_field.extent_decision import ExtentDecision, render_row, probability, CUTOFF


def fixture():
    # Redundant foreground, missing foreground, and two background footprints.
    state = dict(pointer=torch.tensor([0, 2, 3, 5]), index=torch.tensor([0, 1, 2, 0, 3]),
                 alpha=torch.tensor([.9, .8, .9, .7, .9]), atom=torch.zeros(3, dtype=torch.long))
    return ExtentFactors({'views.0.'+k: v for k, v in state.items()}, 4)


def reference(factors, selected, target, positive, negative, confidence, cost):
    member = torch.tensor(selected, dtype=torch.float64); result = .1 * np.dot(cost, selected)
    for vi, f in enumerate(factors.views):
        p = torch.sigmoid(torch.logit(f(member).clamp(1e-6, 1-1e-6)) - torch.logit(torch.tensor(CUTOFF, dtype=torch.float64)))
        y = torch.tensor(target, dtype=torch.float64); pos = torch.tensor(positive); neg = torch.tensor(negative); known = pos | neg
        bce = torch.nn.functional.binary_cross_entropy(p.clamp(1e-5, 1-1e-5), y, reduction='none')
        loss = .5 * ((bce * pos).sum()/max(pos.sum(), 1) + (bce * neg).sum()/max(neg.sum(), 1))
        loss += 1 - (2*(p*y*known).sum()+1)/((p*known).sum()+(y*known).sum()+1)
        result += confidence[vi] / np.count_nonzero(confidence) * float(loss)
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
    f = ExtentFactors({'views.0.'+k: v for k, v in dict(pointer=torch.tensor(ptr), index=torch.tensor(ids), alpha=torch.tensor(alpha), atom=torch.tensor([0])).items()}, 3)
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


def test_extent_field_roundtrip_preserves_parent(tmp_path):
    from test_query_preserving_field import example
    from radio_gs.query_field.storage import load_field, save_field
    field = example(); old = {k: v.clone() for k, v in field.state_dict().items()}
    field.extent_factors = ExtentFactors(fixture().state_dict(), len(field.xyz))
    save_field(field, tmp_path/'field', {})
    restored, manifest = load_field(tmp_path/'field')
    for k, v in old.items(): assert torch.equal(v, restored.state_dict()[k])
    assert manifest['external_scene_dependencies'] == []
    member = torch.ones(len(field.xyz))
    assert torch.equal(restored.extent_factors.views[0](member), field.extent_factors.views[0](member))
