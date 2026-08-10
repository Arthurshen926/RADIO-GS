from __future__ import annotations

from radio_gs.querying import (
    corroborated_scale_aware_native_v3_support_fusion as corroborated,
)
from radio_gs.scripts import (
    materialize_lerf_o1_native_v3_corroborated_fusion as materializer,
)


def test_corroborated_materializer_contract_has_no_new_numeric_constant() -> None:
    assert materializer.O1_RESULT_CONTRACTS == {
        "figurines_oracle_matrix",
        "streaming_source_only",
    }
    contract = corroborated.readout_contract()
    assert contract["invariants"]["new_tunable_constants"] is False
    assert contract["semantic_boundary"] == 0.6
