import numpy as np
import torch
from radio_gs.query_field.extent_factors import ExtentFactors
from radio_gs.query_field.extent_decision import ExtentDecision
from radio_gs.query_field.extent_decision_linked import LinkedExtentDecision


def test_reference_and_linked_all_decisions_and_traces_exact():
    rng = np.random.default_rng(20261011)
    for case in range(24):
        state = {}; count = 17
        for vi in range(3):
            pointer = [0]; ids = []; values = []
            for row in range(19):
                hit = rng.choice(count, rng.integers(0, count+1), replace=False)
                ids.extend(hit.tolist()); values.extend(rng.uniform(.004, 1, len(hit)).tolist()); pointer.append(len(ids))
            v = dict(pointer=torch.tensor(pointer), index=torch.tensor(ids, dtype=torch.long),
                alpha=torch.tensor(values), visibility=torch.tensor(values), atom=torch.zeros(19, dtype=torch.long))
            state.update({f'views.{vi}.'+k: t for k,t in v.items()})
        f = ExtentFactors(state, count)
        initial = rng.random(count) < .4; target = rng.random(57)
        positive = target > .7; negative = target < .3
        confidence = np.array([.3,.8,0. if case%2 else .4]); cost = rng.random(count)/count
        a, ar = ExtentDecision(f,count).solve(initial,target,positive,negative,confidence,cost)
        b, br = LinkedExtentDecision(f,count).solve(initial,target,positive,negative,confidence,cost)
        assert np.array_equal(a,b) and ar == br
