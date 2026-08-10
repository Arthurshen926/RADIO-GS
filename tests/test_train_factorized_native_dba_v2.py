from __future__ import annotations

from copy import deepcopy

import pytest

from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_dba_v1 as dba_v1,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_dba_v2 as dba_v2,
)


SHA = "a" * 64


def _authority() -> dict:
    record = {"path": "/frozen", "sha256": SHA}
    return {
        "schema": dba_v2.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": dba_v2.SCHEMA_VERSION,
        "status": (
            "authorized_source_only_precision_ranking_dba_v2_"
            "exact4train_2validation"
        ),
        "implementation": record,
        "implementation_dependencies": {
            name: record for name in dba_v2._DEPENDENCY_PATHS
        },
        "design_preregistration": record,
        "base_dba_v1_execution_authority": record,
        "training_contract_sha256": dba_v2.TRAINING_CONTRACT_SHA256,
        "training_output": "/frozen/model.pt",
        "training_authorized": True,
        "target_execution_authorized": False,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "source_access": dba_v2.source_access(),
    }


def test_contract_replaces_fully_balanced_negatives_with_precision_tail() -> None:
    contract = dba_v2.training_contract()
    objective = contract["objective"]
    assert objective["minimum_precision"] == 0.25
    assert objective["hard_negatives_per_positive"] == 3
    assert objective["global_order"]["pair_cap"] == 4096
    assert objective["boundary_auxiliary_weight_on_complete_dba_v2_loss"] == 0.25
    assert contract["optimizer"] == dba_v1.training_contract()["optimizer"]
    inherited = deepcopy(dba_v1.training_contract()["promotion"])
    observed = deepcopy(contract["promotion"])
    assert observed.pop("inherited_unchanged_from_dba_v1") is True
    assert observed == inherited


def test_authority_remains_source_only_and_binds_design_and_v1_chain() -> None:
    dba_v2.validate_execution_authority_header(_authority())
    for field in (
        "target_execution_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
    ):
        changed = deepcopy(_authority())
        changed[field] = True
        with pytest.raises(ValueError, match="header differs"):
            dba_v2.validate_execution_authority_header(changed)


def test_synthetic_schedule_and_gate_are_inherited_unchanged() -> None:
    result = dba_v2.synthetic_dry_run()
    assert result["hard_negatives_per_positive"] == 3
    assert result["complete_rows_per_scene"] == 4096
    assert result["evaluation_steps"] == list(range(0, 65, 8))
    assert result["promotion_gate_bitwise_inherited_from_dba_v1"] is True
    assert result["target_query_or_benchmark_opened"] is False
