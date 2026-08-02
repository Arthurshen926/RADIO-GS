#!/usr/bin/env python3
"""Bind an NVOS forward-Beta candidate to fail-closed protocol authority.

This CPU-only authority separates two positive bindings from the external
LUDVIG comparator fence:

* the 2026-07-16 RADIO-GS parent-method/evaluation freeze is lineage only;
* ``nvos_strict_unseen_v1`` is the candidate benchmark protocol;
* the 2026-08-01 LUDVIG row is comparator-only and can never become the
  candidate task or registry row.

No caller supplies an ``exact`` flag.  Strict protocol exactness is derived
from the complete scoring contract, so a centered Beta posterior remains a
different representation even when its binary decision is mathematically
equivalent to thresholding a zero-centered margin.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from radio_gs.scripts.bind_evaluation_protocol_freeze import (
    write_binding_receipt,
)
from radio_gs.scripts.validate_evaluation_protocol_freeze import (
    FreezeError,
    load_and_validate,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "nvos_forward_beta_protocol_authority"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

PARENT_FREEZE_RELATIVE = Path(
    "paper/artifacts/canonical_mpr_v3_evaluation_freeze_20260716.yaml"
)
PARENT_FREEZE_SHA256 = (
    "9ea6fc8d79ee11ae2ccb3b6ef738580983cae406f586db61833182780587c009"
)
PARENT_FREEZE_NAME = "canonical-mpr-v3-evaluation-freeze"
PARENT_BASE_GIT_COMMIT = "2f0e778a4db5da033743118e41c291114a17d0bc"
PARENT_NVOS_SUBCONTRACT_SHA256 = (
    "e520f405e6cad0dc9d49d4f8c80d8a058ff84a86047f21156f51b2e0f0b71d97"
)

PROMPTABLE_REGISTRY_RELATIVE = Path(
    "paper/artifacts/promptable_nvs_protocol_registry.yaml"
)
PROMPTABLE_REGISTRY_SHA256 = (
    "5d1a044513ce2c5d3850dbd95f4a3505c566ae2649e5d89b7e926daf4a568c54"
)
STRICT_PROTOCOL_ID = "nvos_strict_unseen_v1"
STRICT_PROTOCOL_ROW_SHA256 = (
    "91021c9b28df305b7bd52fd3d1c59461f5e2f5bd0d81f659aa5bfaa3639b8d6f"
)

GENERAL_FREEZE_RELATIVE = Path(
    "paper/artifacts/evaluation_protocol_freeze_20260801.yaml"
)
GENERAL_FREEZE_SHA256 = (
    "af91f0861d3a15354063579e78f64898801c41f2543d1cf9b352a0a123820916"
)
GENERAL_FREEZE_ID = "evaluation_protocols_20260801_v1"
LUDVIG_TASK_ID = "spatial_nvos_ludvig"
LUDVIG_REGISTRY_ROW = (
    "nvos_ludvig_released_all_view_full8_3seed_exact_20260731"
)
LUDVIG_PROMPTABLE_ROW = (
    "ludvig_nvos_released_all_view_full8_exact_3seed_v1"
)
LUDVIG_GENERAL_TASK_SHA256 = (
    "2ee4ad0eb21886d10d3421b459ed69eae9b8ea92b0fc3a08f7e07b71ee691514"
)
LUDVIG_PROMPTABLE_ROW_SHA256 = (
    "c5cc3fce9f5cbc7f71501d19fce3aa59e3ed95d751da76ccef68ba5676580d83"
)
PRIMARY_REGISTRY_RELATIVE = Path(
    "paper/artifacts/evaluation_protocol_registry_20260731.yaml"
)
PRIMARY_REGISTRY_SHA256 = (
    "ed394c6a5dfd03143805de512169aa01bd684b9546561f321c53211a135da2c6"
)
LUDVIG_PRIMARY_ROW_SHA256 = (
    "1cee63751c9150ccb176e908ad83b782d0d4fb36b3ac33de698e22248a163654"
)

STRICT_TASKS = (
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
)
STRICT_SCORING_CONTRACT = {
    "score_semantics": "continuous_cosine_margin",
    "prediction_representation": "continuous_cosine_margin",
    "threshold": {"comparison": "greater_or_equal", "value": 0.0},
    "resize": "nearest",
}


class AuthorityError(ValueError):
    """Raised when protocol provenance is incomplete, altered, or ambiguous."""


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuthorityError(f"{label} must be a mapping")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def _resolve(root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    candidate = path if path.is_absolute() else root / path
    # Keep the lexical path. Resolving here would silently follow a symlink
    # before the no-follow descriptor walk below can reject it.
    return Path(os.path.abspath(os.fspath(candidate)))


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_nofollow(path: Path, *, label: str) -> int:
    """Open one absolute path without following any component symlink."""

    if not path.is_absolute():
        raise AuthorityError(f"{label} path must be absolute")
    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise AuthorityError("platform lacks fail-closed no-follow file flags")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    parts = path.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise AuthorityError(f"{label} path is invalid: {path}")

    directory_descriptor = os.open(os.sep, directory_flags)
    try:
        for component in parts[1:-1]:
            try:
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                if error.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise AuthorityError(
                        f"{label} refuses symlink or unsafe directory component: "
                        f"{path}"
                    ) from error
                raise AuthorityError(f"cannot open {label}: {path}: {error}") from error
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        try:
            return os.open(parts[-1], file_flags, dir_fd=directory_descriptor)
        except OSError as error:
            if error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise AuthorityError(
                    f"{label} refuses symlink or unsafe final component: {path}"
                ) from error
            raise AuthorityError(f"cannot open {label}: {path}: {error}") from error
    finally:
        os.close(directory_descriptor)


def _read_stable_bytes(path: Path, *, label: str) -> bytes:
    """Read/hash source bytes once and prove descriptor/path identity stability."""

    descriptor = _open_nofollow(path, label=label)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AuthorityError(f"{label} must be a regular file: {path}")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after):
        raise AuthorityError(f"{label} changed while it was being read: {path}")
    encoded = b"".join(blocks)
    if len(encoded) != after.st_size:
        raise AuthorityError(f"{label} size changed while it was being read: {path}")

    # Detect atomic replacement of the lexical path while the original
    # descriptor was open. This second open reads no content.
    confirmation_descriptor = _open_nofollow(path, label=label)
    try:
        confirmation = os.fstat(confirmation_descriptor)
    finally:
        os.close(confirmation_descriptor)
    if _identity(after) != _identity(confirmation):
        raise AuthorityError(f"{label} path changed while it was being read: {path}")
    return encoded


def _load_exact_yaml(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[Path, Mapping[str, Any]]:
    lexical = _resolve(Path(os.sep), path)
    encoded = _read_stable_bytes(lexical, label=label)
    actual_sha256 = hashlib.sha256(encoded).hexdigest()
    if actual_sha256 != expected_sha256:
        raise AuthorityError(
            f"{label} file SHA256 drifted: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    try:
        decoded = encoded.decode("utf-8")
        payload = yaml.safe_load(decoded)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise AuthorityError(f"{label} is not valid UTF-8 YAML") from error
    return lexical, _mapping(payload, label)


def _validate_parent_freeze_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(payload.get("schema_version") == 1, "parent freeze schema differs")
    _require(payload.get("name") == PARENT_FREEZE_NAME, "parent freeze name differs")
    _require(
        payload.get("base_git_commit") == PARENT_BASE_GIT_COMMIT,
        "parent freeze base commit differs",
    )
    policy = _mapping(payload.get("policy"), "parent freeze policy")
    _require(
        policy.get("method_and_hyperparameters_frozen") is True,
        "parent method freeze flag differs",
    )
    forbidden = policy.get("forbidden_after_freeze")
    _require(isinstance(forbidden, list), "parent forbidden policy must be a list")
    for required in (
        "scene_specific_tuning",
        "benchmark_label_or_mask_calibration",
        "query_set_calibration",
        "test_set_calibration",
    ):
        _require(required in forbidden, f"parent freeze no longer forbids {required}")
    promptable = _mapping(
        payload.get("promptable_reconstruction"),
        "parent promptable reconstruction",
    )
    nvos = _mapping(promptable.get("nvos"), "parent NVOS subcontract")
    _require(
        canonical_json_sha256(nvos) == PARENT_NVOS_SUBCONTRACT_SHA256,
        "parent NVOS subcontract canonical digest drifted",
    )
    _require(nvos.get("scenes") == 8, "parent NVOS scene count differs")
    _require(
        nvos.get("target_rgb_absent_from_geometry_and_feature_extraction") is True,
        "parent NVOS target-RGB exclusion differs",
    )
    return nvos


def _validate_promptable_registry_payload(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    _require(payload.get("schema_version") == 1, "promptable registry schema differs")
    protocols = _mapping(payload.get("protocols"), "promptable registry protocols")
    strict = _mapping(protocols.get(STRICT_PROTOCOL_ID), STRICT_PROTOCOL_ID)
    _require(
        canonical_json_sha256(strict) == STRICT_PROTOCOL_ROW_SHA256,
        "nvos_strict_unseen_v1 canonical row digest drifted",
    )
    _require(strict.get("dataset") == "NVOS", "strict protocol dataset differs")
    _require(
        strict.get("tasks") == list(STRICT_TASKS),
        "strict protocol task cohort differs",
    )
    _require(
        strict.get("prompt")
        == {
            "frame_role": "reference",
            "type": "fixed_positive_and_negative_scribbles",
            "source": "official_nvos_release",
        },
        "strict protocol prompt differs",
    )
    evaluation = _mapping(strict.get("evaluation"), "strict evaluation")
    _require(evaluation.get("frames_per_task") == 1, "strict frame count differs")
    _require(evaluation.get("reference_scored") is False, "reference scoring differs")
    _require(
        evaluation.get("metrics") == ["foreground_iou", "binary_pixel_accuracy"],
        "strict metrics differ",
    )
    _require(
        evaluation.get("aggregation") == "equal_weight_macro_over_8_tasks",
        "strict aggregation differs",
    )
    _require(
        evaluation.get("mask_resize") == STRICT_SCORING_CONTRACT["resize"],
        "strict resize differs",
    )
    _require(
        evaluation.get("prediction_representation")
        == STRICT_SCORING_CONTRACT["prediction_representation"],
        "strict prediction representation differs",
    )
    _require(
        evaluation.get("threshold") == STRICT_SCORING_CONTRACT["threshold"],
        "strict threshold differs",
    )
    visibility = _mapping(strict.get("visibility"), "strict visibility")
    _require(
        visibility.get("target_rgb_allowed_during_field_training") is False,
        "strict field-training visibility differs",
    )
    _require(
        visibility.get("target_rgb_allowed_at_query") is False,
        "strict query visibility differs",
    )
    _require(
        visibility.get("target_mask_use") == "scoring_only",
        "strict target-mask role differs",
    )
    _require(
        visibility.get("sparse_initialization")
        == "drop_any_point_observed_by_target_camera",
        "strict sparse initialization differs",
    )
    calibration = _mapping(strict.get("calibration"), "strict calibration")
    _require(
        calibration.get("target_mask") == "forbidden",
        "target calibration differs",
    )

    ludvig = _mapping(protocols.get(LUDVIG_PROMPTABLE_ROW), LUDVIG_PROMPTABLE_ROW)
    _require(
        canonical_json_sha256(ludvig) == LUDVIG_PROMPTABLE_ROW_SHA256,
        "LUDVIG promptable comparator row digest drifted",
    )
    eligibility = _mapping(ludvig.get("eligibility"), "LUDVIG eligibility")
    _require(
        eligibility.get("strict_unseen_exact_match") is False,
        "LUDVIG promptable row must remain non-strict-unseen",
    )
    return strict, ludvig


def _validate_general_freeze_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(payload.get("freeze_id") == GENERAL_FREEZE_ID, "general freeze id differs")
    tasks = _mapping(payload.get("canonical_tasks"), "general canonical tasks")
    ludvig = _mapping(tasks.get(LUDVIG_TASK_ID), LUDVIG_TASK_ID)
    _require(
        canonical_json_sha256(ludvig) == LUDVIG_GENERAL_TASK_SHA256,
        "general LUDVIG task canonical digest drifted",
    )
    _require(ludvig.get("method") == "LUDVIG-SAM", "general NVOS method differs")
    _require(
        ludvig.get("registry_row") == LUDVIG_REGISTRY_ROW,
        "general LUDVIG primary registry row differs",
    )
    _require(
        ludvig.get("promptable_registry_row") == LUDVIG_PROMPTABLE_ROW,
        "general LUDVIG promptable registry row differs",
    )
    frozen = _mapping(ludvig.get("frozen_protocol"), "general LUDVIG protocol")
    _require(
        frozen.get("strict_unseen_claim") is False,
        "general LUDVIG task must remain non-strict-unseen",
    )
    return ludvig


def _validate_primary_ludvig_row(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluations = _mapping(payload.get("evaluations"), "primary registry evaluations")
    row = _mapping(evaluations.get(LUDVIG_REGISTRY_ROW), LUDVIG_REGISTRY_ROW)
    _require(
        canonical_json_sha256(row) == LUDVIG_PRIMARY_ROW_SHA256,
        "LUDVIG primary comparator row canonical digest drifted",
    )
    _require(row.get("method") == "LUDVIG-SAM", "LUDVIG primary method differs")
    protocol = _mapping(row.get("protocol"), "LUDVIG primary protocol")
    _require(
        protocol.get("strict_unseen_exact_match") is False,
        "LUDVIG primary row must remain non-strict-unseen",
    )
    artifacts = _mapping(row.get("artifacts"), "LUDVIG primary artifacts")
    _require(
        artifacts.get("eligible_for_strict_unseen_claim") is False,
        "LUDVIG primary artifact eligibility must remain non-strict-unseen",
    )
    return row


def scoring_exactness(scoring_contract: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Derive strict-row exactness; callers cannot override the decision."""

    scoring = dict(scoring_contract)
    blockers: list[str] = []
    for field in ("score_semantics", "prediction_representation", "resize"):
        if scoring.get(field) != STRICT_SCORING_CONTRACT[field]:
            blockers.append(f"{field}_differs")
    threshold = scoring.get("threshold")
    expected_threshold = STRICT_SCORING_CONTRACT["threshold"]
    if not isinstance(threshold, Mapping):
        blockers.append("threshold_differs")
    else:
        comparison = threshold.get("comparison")
        try:
            value = float(threshold.get("value"))
        except (TypeError, ValueError):
            value = math.nan
        if (
            comparison != expected_threshold["comparison"]
            or not math.isfinite(value)
            or value != float(expected_threshold["value"])
        ):
            blockers.append("threshold_differs")
    return not blockers, blockers


def _validated_scoring_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    scoring = dict(value)
    required = {
        "score_semantics",
        "prediction_representation",
        "threshold",
        "resize",
    }
    _require(set(scoring) == required, "scoring contract fields differ")
    _require(
        isinstance(scoring["score_semantics"], str)
        and bool(scoring["score_semantics"]),
        "score_semantics must be non-empty",
    )
    _require(
        isinstance(scoring["prediction_representation"], str)
        and bool(scoring["prediction_representation"]),
        "prediction_representation must be non-empty",
    )
    _require(
        isinstance(scoring["resize"], str) and bool(scoring["resize"]),
        "resize must be non-empty",
    )
    threshold = _mapping(scoring["threshold"], "scoring threshold")
    _require(
        set(threshold) == {"comparison", "value"},
        "scoring threshold fields differ",
    )
    _require(
        isinstance(threshold.get("comparison"), str)
        and bool(threshold.get("comparison")),
        "threshold comparison must be non-empty",
    )
    try:
        threshold_value = float(threshold.get("value"))
    except (TypeError, ValueError) as error:
        raise AuthorityError("threshold value must be finite") from error
    _require(math.isfinite(threshold_value), "threshold value must be finite")
    scoring["threshold"] = {
        "comparison": str(threshold["comparison"]),
        "value": threshold_value,
    }
    return scoring


def build_authority(
    *,
    candidate_method_sha256: str,
    scoring_contract: Mapping[str, Any],
    repo_root: str | Path | None = None,
    parent_freeze: str | Path = PARENT_FREEZE_RELATIVE,
    promptable_registry: str | Path = PROMPTABLE_REGISTRY_RELATIVE,
    general_freeze: str | Path = GENERAL_FREEZE_RELATIVE,
    primary_registry: str | Path = PRIMARY_REGISTRY_RELATIVE,
) -> dict[str, Any]:
    """Validate all authority sources and build immutable provenance data."""

    if not isinstance(candidate_method_sha256, str) or not SHA256_RE.fullmatch(
        candidate_method_sha256
    ):
        raise AuthorityError("candidate method SHA256 must be 64 lowercase hex")
    scoring = _validated_scoring_contract(scoring_contract)
    root = (
        _resolve(Path.cwd(), repo_root)
        if repo_root is not None
        else _resolve(Path.cwd(), Path(__file__)).parents[2]
    )

    parent_path, parent_payload = _load_exact_yaml(
        _resolve(root, parent_freeze),
        expected_sha256=PARENT_FREEZE_SHA256,
        label="RADIO-GS parent evaluation freeze",
    )
    _validate_parent_freeze_payload(parent_payload)

    promptable_path, promptable_payload = _load_exact_yaml(
        _resolve(root, promptable_registry),
        expected_sha256=PROMPTABLE_REGISTRY_SHA256,
        label="promptable protocol registry",
    )
    _validate_promptable_registry_payload(promptable_payload)

    general_path, general_payload = _load_exact_yaml(
        _resolve(root, general_freeze),
        expected_sha256=GENERAL_FREEZE_SHA256,
        label="general evaluation protocol freeze",
    )
    try:
        validated_general = load_and_validate(
            general_path,
            root=root,
            verify_hashes=True,
        )
    except FreezeError as error:
        raise AuthorityError(f"general freeze validation failed: {error}") from error
    rechecked_general_path, rechecked_general = _load_exact_yaml(
        general_path,
        expected_sha256=GENERAL_FREEZE_SHA256,
        label="general evaluation protocol freeze",
    )
    _require(
        rechecked_general_path == general_path,
        "general freeze path changed during validation",
    )
    _require(
        validated_general == general_payload == rechecked_general,
        "general freeze payload changed during validation",
    )
    rechecked_promptable_path, rechecked_promptable = _load_exact_yaml(
        promptable_path,
        expected_sha256=PROMPTABLE_REGISTRY_SHA256,
        label="promptable protocol registry",
    )
    _require(
        rechecked_promptable_path == promptable_path
        and rechecked_promptable == promptable_payload,
        "promptable registry changed during general freeze validation",
    )
    _validate_general_freeze_payload(general_payload)

    primary_path, primary_payload = _load_exact_yaml(
        _resolve(root, primary_registry),
        expected_sha256=PRIMARY_REGISTRY_SHA256,
        label="primary evaluation protocol registry",
    )
    _validate_primary_ludvig_row(primary_payload)

    exact, blockers = scoring_exactness(scoring)
    provenance = {
        "schema_version": 1,
        "radio_gs_parent_evaluation_freeze": {
            "path": str(parent_path),
            "file_sha256": PARENT_FREEZE_SHA256,
            "freeze_name": PARENT_FREEZE_NAME,
            "base_git_commit": PARENT_BASE_GIT_COMMIT,
            "nvos_subcontract_canonical_json_sha256": (
                PARENT_NVOS_SUBCONTRACT_SHA256
            ),
            "binding_role": "parent_field_and_safety_lineage",
            "parent_method_exact_match": False,
            "candidate_method_contract_sha256": candidate_method_sha256,
        },
        "strict_unseen_benchmark_registry": {
            "path": str(promptable_path),
            "file_sha256": PROMPTABLE_REGISTRY_SHA256,
            "protocol_id": STRICT_PROTOCOL_ID,
            "protocol_row_canonical_json_sha256": STRICT_PROTOCOL_ROW_SHA256,
            "exact_match": exact,
            "exact_match_blockers": blockers,
        },
    }
    authority = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "validated_and_bound",
        "candidate": {
            "method_family": "RADIO-GS",
            "method_contract_sha256": candidate_method_sha256,
            "parent_method_exact_match": False,
        },
        "scoring_contract": scoring,
        "strict_unseen_protocol_exact_match": exact,
        "strict_unseen_exact_match_blockers": blockers,
        "protocol_provenance": provenance,
        "protocol_provenance_sha256": canonical_json_sha256(provenance),
        "external_comparator_provenance": {
            "path": str(general_path),
            "file_sha256": GENERAL_FREEZE_SHA256,
            "freeze_id": GENERAL_FREEZE_ID,
            "binding_role": "external_method_comparator_only",
            "candidate_binding": {
                "canonical_task_id": None,
                "registry_row": None,
                "promptable_registry_row": None,
            },
            "excluded_from_candidate_authority": {
                "canonical_task_id": LUDVIG_TASK_ID,
                "canonical_task_canonical_json_sha256": (
                    LUDVIG_GENERAL_TASK_SHA256
                ),
                "registry_path": str(primary_path),
                "registry_file_sha256": PRIMARY_REGISTRY_SHA256,
                "registry_row": LUDVIG_REGISTRY_ROW,
                "registry_row_canonical_json_sha256": (
                    LUDVIG_PRIMARY_ROW_SHA256
                ),
                "promptable_registry_row": LUDVIG_PROMPTABLE_ROW,
                "promptable_registry_row_canonical_json_sha256": (
                    LUDVIG_PROMPTABLE_ROW_SHA256
                ),
                "strict_unseen_exact_match": False,
            },
        },
    }
    validate_authority_payload(authority)
    return authority


def validate_authority_payload(payload: Mapping[str, Any]) -> None:
    """Validate an authority fragment before embedding or atomic publication."""

    _require(
        payload.get("schema_version") == SCHEMA_VERSION,
        "authority schema differs",
    )
    _require(payload.get("artifact_type") == ARTIFACT_TYPE, "authority type differs")
    _require(payload.get("status") == "validated_and_bound", "authority status differs")
    candidate = _mapping(payload.get("candidate"), "authority candidate")
    method_sha = candidate.get("method_contract_sha256")
    _require(
        candidate.get("method_family") == "RADIO-GS",
        "candidate method family differs",
    )
    _require(
        isinstance(method_sha, str) and bool(SHA256_RE.fullmatch(method_sha)),
        "candidate method SHA256 differs",
    )
    _require(
        candidate.get("parent_method_exact_match") is False,
        "parent_method_exact_match must remain false",
    )
    scoring = _validated_scoring_contract(
        _mapping(payload.get("scoring_contract"), "authority scoring contract")
    )
    exact, blockers = scoring_exactness(scoring)
    _require(
        payload.get("strict_unseen_protocol_exact_match") is exact,
        "strict-unseen exactness was not derived from scoring contract",
    )
    _require(
        payload.get("strict_unseen_exact_match_blockers") == blockers,
        "strict-unseen exactness blockers differ",
    )
    provenance = _mapping(payload.get("protocol_provenance"), "protocol provenance")
    _require(
        payload.get("protocol_provenance_sha256")
        == canonical_json_sha256(provenance),
        "protocol provenance digest differs",
    )
    parent = _mapping(
        provenance.get("radio_gs_parent_evaluation_freeze"),
        "parent provenance",
    )
    _require(
        parent.get("file_sha256") == PARENT_FREEZE_SHA256
        and parent.get("freeze_name") == PARENT_FREEZE_NAME
        and parent.get("base_git_commit") == PARENT_BASE_GIT_COMMIT
        and parent.get("nvos_subcontract_canonical_json_sha256")
        == PARENT_NVOS_SUBCONTRACT_SHA256
        and parent.get("binding_role") == "parent_field_and_safety_lineage",
        "parent provenance authority differs",
    )
    _require(
        parent.get("parent_method_exact_match") is False,
        "parent provenance exact-match flag must remain false",
    )
    _require(
        parent.get("candidate_method_contract_sha256") == method_sha,
        "parent provenance candidate SHA differs",
    )
    strict = _mapping(
        provenance.get("strict_unseen_benchmark_registry"),
        "strict provenance",
    )
    _require(
        strict.get("file_sha256") == PROMPTABLE_REGISTRY_SHA256
        and strict.get("protocol_id") == STRICT_PROTOCOL_ID
        and strict.get("protocol_row_canonical_json_sha256")
        == STRICT_PROTOCOL_ROW_SHA256,
        "strict protocol authority differs",
    )
    _require(strict.get("exact_match") is exact, "strict provenance exactness differs")
    _require(strict.get("exact_match_blockers") == blockers, "strict blockers differ")
    comparator = _mapping(
        payload.get("external_comparator_provenance"),
        "external comparator provenance",
    )
    _require(
        comparator.get("file_sha256") == GENERAL_FREEZE_SHA256
        and comparator.get("freeze_id") == GENERAL_FREEZE_ID
        and comparator.get("binding_role") == "external_method_comparator_only",
        "LUDVIG binding role differs",
    )
    candidate_binding = _mapping(
        comparator.get("candidate_binding"),
        "external comparator candidate binding",
    )
    _require(
        candidate_binding
        == {
            "canonical_task_id": None,
            "registry_row": None,
            "promptable_registry_row": None,
        },
        "LUDVIG comparator cannot be bound as the RADIO-GS candidate",
    )
    excluded = _mapping(
        comparator.get("excluded_from_candidate_authority"),
        "excluded LUDVIG authority",
    )
    _require(
        excluded.get("canonical_task_id") == LUDVIG_TASK_ID
        and excluded.get("canonical_task_canonical_json_sha256")
        == LUDVIG_GENERAL_TASK_SHA256
        and excluded.get("registry_row") == LUDVIG_REGISTRY_ROW
        and excluded.get("registry_file_sha256") == PRIMARY_REGISTRY_SHA256
        and excluded.get("registry_row_canonical_json_sha256")
        == LUDVIG_PRIMARY_ROW_SHA256
        and excluded.get("promptable_registry_row") == LUDVIG_PROMPTABLE_ROW
        and excluded.get("promptable_registry_row_canonical_json_sha256")
        == LUDVIG_PROMPTABLE_ROW_SHA256
        and excluded.get("strict_unseen_exact_match") is False,
        "LUDVIG comparator fence differs",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--candidate-method-sha256", required=True)
    parser.add_argument("--score-semantics", required=True)
    parser.add_argument("--prediction-representation", required=True)
    parser.add_argument("--threshold-comparison", required=True)
    parser.add_argument("--threshold-value", type=float, required=True)
    parser.add_argument("--resize", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    authority = build_authority(
        candidate_method_sha256=args.candidate_method_sha256,
        scoring_contract={
            "score_semantics": args.score_semantics,
            "prediction_representation": args.prediction_representation,
            "threshold": {
                "comparison": args.threshold_comparison,
                "value": args.threshold_value,
            },
            "resize": args.resize,
        },
        repo_root=args.repo_root,
    )
    validate_authority_payload(authority)
    write_binding_receipt(args.output, authority)
    print(
        f"bound {STRICT_PROTOCOL_ID}; "
        f"exact={authority['strict_unseen_protocol_exact_match']} at {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
