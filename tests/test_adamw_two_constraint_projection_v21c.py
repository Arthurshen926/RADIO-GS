from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from radio_gs.optimization import adamw_two_constraint_projection_v21c as method


@pytest.mark.parametrize(
    ("candidate", "absolute", "pairwise", "expected"),
    [
        ([1.0, 2.0], [1.0, 0.0], [0.0, 1.0], [1.0, 2.0]),
        ([-1.0, 2.0], [1.0, 0.0], [0.0, 1.0], [0.0, 2.0]),
        ([-1.0, -2.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]),
    ],
)
def test_projection_known_solutions_and_kkt(
    candidate, absolute, pairwise, expected
) -> None:
    projected, evidence = method.project_two_halfspaces(
        torch.tensor(candidate, dtype=torch.float64),
        torch.tensor(absolute, dtype=torch.float64),
        torch.tensor(pairwise, dtype=torch.float64),
    )
    assert torch.allclose(projected, torch.tensor(expected, dtype=torch.float64))
    assert evidence["kkt"]["passed"] is True
    assert evidence["projected_dot"]["absolute"] >= -1e-10
    assert evidence["projected_dot"]["pairwise"] >= -1e-10


def test_projection_handles_collinear_redundant_constraints() -> None:
    candidate = torch.tensor([-2.0, 3.0], dtype=torch.float64)
    projected, evidence = method.project_two_halfspaces(
        candidate,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([2.0, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(projected, torch.tensor([0.0, 3.0], dtype=torch.float64))
    assert evidence["kkt"]["passed"] is True


def test_adamw_prediction_matches_real_step_with_moments_and_weight_decay() -> None:
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.GELU(), torch.nn.Linear(4, 2)
    ).double()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, betas=(0.81, 0.93), eps=1e-7, weight_decay=0.2
    )
    named = method.trainable_named_parameters(model)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randn(5, 3, dtype=torch.float64)).square().mean()
        loss.backward()
        before = method.flatten_parameter_values(named).clone()
        predicted = method.predict_adamw_descent_direction(optimizer, named)
        optimizer.step()
        actual = before - method.flatten_parameter_values(named)
        assert torch.allclose(predicted, actual, rtol=1e-10, atol=1e-12)


def test_projected_commit_preserves_ordinary_adamw_moment_transition() -> None:
    parameter = torch.nn.Parameter(torch.tensor([-1.0, -2.0], dtype=torch.float64))
    ordinary_parameter = torch.nn.Parameter(parameter.detach().clone())
    projected_optimizer = torch.optim.AdamW(
        [parameter], lr=0.1, betas=(0.5, 0.75), weight_decay=0.1
    )
    ordinary_optimizer = torch.optim.AdamW(
        [ordinary_parameter], lr=0.1, betas=(0.5, 0.75), weight_decay=0.1
    )
    gradient = torch.tensor([1.0, 1.0], dtype=torch.float64)
    parameter.grad = gradient.clone()
    ordinary_parameter.grad = gradient.clone()
    evidence = method.commit_projected_adamw_step(
        projected_optimizer,
        (("p", parameter),),
        torch.tensor([-1.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, -1.0], dtype=torch.float64),
    )
    ordinary_optimizer.step()
    projected_state = projected_optimizer.state[parameter]
    ordinary_state = ordinary_optimizer.state[ordinary_parameter]
    assert torch.equal(projected_state["step"], ordinary_state["step"])
    assert torch.equal(projected_state["exp_avg"], ordinary_state["exp_avg"])
    assert torch.equal(projected_state["exp_avg_sq"], ordinary_state["exp_avg_sq"])
    assert evidence["kkt"]["passed"] is True
    assert evidence["projected_dot"]["absolute"] >= -1e-10
    assert evidence["projected_dot"]["pairwise"] >= -1e-10


def test_parameter_subset_manifest_is_ordered_and_tamper_evident() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    named = method.trainable_named_parameters(model)
    manifest = method.parameter_subset_manifest(named)
    assert [name for name, _ in named] == sorted(name for name, _ in named)
    assert manifest["vector_numel"] == sum(parameter.numel() for _, parameter in named)
    changed = deepcopy(manifest["parameter_records"])
    changed[0]["shape"] = [999]
    assert method.canonical_json_sha256(changed) != manifest[
        "parameter_records_sha256"
    ]
