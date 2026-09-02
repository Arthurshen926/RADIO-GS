import pytest
import torch

from radio_gs.v4.object_memory import SparseObjectAssignments
from radio_gs.v4.query import QueryPacket, QuerySelectionMode


@pytest.mark.parametrize("mode", list(QuerySelectionMode))
def test_query_packet_accepts_only_declared_selection_modes(mode):
    assert QueryPacket(mode).selection_mode is mode
    assert QueryPacket(mode.value).selection_mode is mode


@pytest.mark.parametrize(
    "invalid",
    ["single", "single_instance ", "SINGLE_INSTANCE", "", 1, None],
)
def test_query_packet_rejects_aliases_and_untyped_selection_modes(invalid):
    with pytest.raises((TypeError, ValueError), match="selection_mode"):
        QueryPacket(invalid)


def _assignments() -> SparseObjectAssignments:
    return SparseObjectAssignments(
        token_ids=torch.tensor([[0, 1], [0, 1], [1, -1]]),
        weights=torch.tensor([[0.4, 0.3], [0.8, 0.1], [0.5, 0.0]]),
        unknown_weight=torch.tensor([0.3, 0.1, 0.5]),
        num_tokens=2,
    )


def test_single_instance_uses_full_token_plus_null_simplex_and_mixture_sum():
    assignments = _assignments()
    result = assignments.element_posterior(
        QueryPacket("single_instance"),
        torch.tensor([0.6, 0.3]),
        null_probability=0.1,
    )

    # Element 0 deliberately has two retained contributions: 0.4*0.6 + 0.3*0.3.
    # A max compositor would incorrectly return 0.24 instead of 0.33.
    assert result.foreground.tolist() == pytest.approx([0.33, 0.51, 0.15])
    assert result.assignment_unknown.tolist() == pytest.approx([0.27, 0.09, 0.45])
    assert result.query_null_probability == pytest.approx(0.1)
    assert result.selection_mode is QuerySelectionMode.SINGLE_INSTANCE


def test_single_instance_requires_explicit_null_and_global_simplex():
    assignments = _assignments()
    query = QueryPacket("single_instance")
    with pytest.raises(ValueError, match="explicit null_probability"):
        assignments.element_posterior(query, torch.tensor([0.6, 0.3]))
    with pytest.raises(ValueError, match="simplex"):
        assignments.element_posterior(
            query,
            torch.tensor([0.6, 0.3]),
            null_probability=0.2,
        )
    with pytest.raises(ValueError, match="one finite scalar"):
        assignments.element_posterior(
            query,
            torch.tensor([0.6, 0.3]),
            null_probability=torch.tensor([0.1]),
        )


def test_multi_instance_accepts_independent_probabilities_and_uses_same_mixture():
    assignments = _assignments()
    result = assignments.element_posterior(
        QueryPacket("multi_instance"),
        torch.tensor([0.8, 0.7]),
    )

    assert result.foreground.tolist() == pytest.approx([0.53, 0.71, 0.35])
    # Independent Bernoulli no-match probability: (1 - .8) * (1 - .7) = .06.
    assert result.query_null_probability == pytest.approx(0.06)
    assert result.assignment_unknown.tolist() == pytest.approx([0.282, 0.094, 0.47])
    assert result.selection_mode is QuerySelectionMode.MULTI_INSTANCE


def test_multi_instance_forbids_a_second_incompatible_null_contract():
    with pytest.raises(ValueError, match="must not be supplied"):
        _assignments().element_posterior(
            QueryPacket("multi_instance"),
            torch.tensor([0.8, 0.7]),
            null_probability=0.1,
        )


def test_local_semantic_fails_closed_on_object_codebook():
    with pytest.raises(ValueError, match="local surface semantic memory"):
        _assignments().element_posterior(
            QueryPacket("local_semantic"),
            torch.tensor([0.8, 0.7]),
        )


@pytest.mark.parametrize(
    "invalid",
    [torch.tensor([0.1]), torch.tensor([1.1, 0.0]), torch.tensor([float("nan"), 0.0])],
)
def test_all_object_selection_modes_validate_token_probability(invalid):
    with pytest.raises(ValueError, match="token_probability|token probabilities"):
        _assignments().element_posterior(QueryPacket("multi_instance"), invalid)


def test_mixture_sum_is_public_cardinality_independent_primitive():
    assert _assignments().mixture_sum(torch.tensor([0.25, 0.75])).tolist() == pytest.approx(
        [0.325, 0.275, 0.375]
    )
