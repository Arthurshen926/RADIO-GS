from __future__ import annotations

from copy import deepcopy

import pytest

from radio_gs.scripts import (
    build_surface_region_v21a_rescue_execution_authority as builder,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21a_rescue as rescue,
)
from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json
from test_train_surface_region_typed_context_response_listwise_v21_pilot import (
    _authority as baseline_authority,
)


def test_builder_projects_same_assets_but_binds_independent_implementation(
    tmp_path,
) -> None:
    parent_path = write_frozen_json(tmp_path / "parent.json", baseline_authority())
    authority = builder.build(
        parent_path, expected_sha256=sha256_file(parent_path)
    )
    assert authority["schema"] == rescue.EXECUTION_AUTHORITY_SCHEMA
    assert authority["base_v21_asset_execution_authority"] == {
        "path": str(parent_path),
        "sha256": sha256_file(parent_path),
    }
    assert authority["implementation"] == rescue._implementation_records()[
        "implementation"
    ]
    assert authority["implementation"] != authority["parent_implementation"]
    assert authority["source_train"] == baseline_authority()["source_train"]


def test_builder_is_no_clobber_and_authority_rejects_baseline_implementation(
    tmp_path,
) -> None:
    parent_path = write_frozen_json(tmp_path / "parent.json", baseline_authority())
    output = tmp_path / "v21a.json"
    builder.write_authority(
        parent_path,
        expected_sha256=sha256_file(parent_path),
        output=output,
    )
    with pytest.raises(FileExistsError):
        builder.write_authority(
            parent_path,
            expected_sha256=sha256_file(parent_path),
            output=output,
        )
    tampered = deepcopy(
        builder.build(parent_path, expected_sha256=sha256_file(parent_path))
    )
    tampered["implementation"] = tampered["parent_implementation"]
    with pytest.raises(ValueError, match="must not reuse"):
        rescue.validate_execution_authority(tampered)


def test_builder_cli_has_separate_build_and_full_validate_commands() -> None:
    choices = builder.build_parser()._subparsers._group_actions[0].choices
    assert set(choices) == {"build", "validate"}
