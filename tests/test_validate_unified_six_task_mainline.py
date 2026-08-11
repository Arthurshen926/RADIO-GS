from __future__ import annotations

from pathlib import Path

import yaml

from radio_gs.scripts.validate_unified_six_task_mainline import validate_registry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "paper/artifacts/unified_six_task_mainline_v1.yaml"
REGISTRY_V2 = ROOT / "paper/artifacts/unified_six_task_single_radio_mainline_v2.yaml"


def test_repository_registry_is_valid() -> None:
    assert validate_registry(REGISTRY, root=ROOT) == []


def test_single_radio_v2_repository_registry_is_valid() -> None:
    assert validate_registry(REGISTRY_V2, root=ROOT) == []


def test_unified_claim_fails_closed_while_task_gate_is_open(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    payload["claim_boundary"]["unified_paper_claim_eligible"] = True
    candidate = tmp_path / "registry.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    issues = validate_registry(candidate, root=ROOT)
    assert any("while any task gate is open" in issue for issue in issues)


def test_strict_row_cannot_use_target_rgb(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    payload["tasks"]["nvos"]["current_rows"][0][
        "target_rgb_visible_to_method"
    ] = True
    candidate = tmp_path / "registry.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    issues = validate_registry(candidate, root=ROOT)
    assert any("strict rows must set target_rgb_visible_to_method=false" in issue for issue in issues)


def test_oracle_row_cannot_be_primary(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    row = payload["tasks"]["lerf2d"]["current_rows"][0]
    row["track"] = "oracle_diagnostic"
    row["paper_role"] = "primary"
    candidate = tmp_path / "registry.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    issues = validate_registry(candidate, root=ROOT)
    assert any("cannot promote an oracle row" in issue for issue in issues)


def test_v2_primary_cannot_read_source_rgb_at_query_time(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY_V2.read_text(encoding="utf-8"))
    payload["tasks"]["nvos"]["current_rows"][0][
        "query_time_source_rgb_visible_to_method"
    ] = True
    candidate = tmp_path / "registry-v2.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    issues = validate_registry(candidate, root=ROOT)
    assert any("prohibit query-time source RGB" in issue for issue in issues)


def test_v2_primary_cannot_store_sam_or_teacher_payload(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY_V2.read_text(encoding="utf-8"))
    row = payload["tasks"]["lerf2d"]["current_rows"][0]
    row["persistent_semantic_feature"] = "radio_plus_sam"
    row["training_teacher_payload_in_checkpoint"] = True
    candidate = tmp_path / "registry-v2.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    issues = validate_registry(candidate, root=ROOT)
    assert any("store only RADIO" in issue for issue in issues)
    assert any("exclude teacher payloads" in issue for issue in issues)


def test_v2_contract_cannot_enable_target_rgb(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY_V2.read_text(encoding="utf-8"))
    payload["mainline_contract"]["query_time"]["target_rgb_allowed"] = True
    candidate = tmp_path / "registry-v2.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    issues = validate_registry(candidate, root=ROOT)
    assert any("target_rgb_allowed=false" in issue for issue in issues)


def test_v2_source_sam_upgrade_cannot_persist_teacher_or_use_rgb_at_eval(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(REGISTRY_V2.read_text(encoding="utf-8"))
    payload["shared_architecture_target"]["build_time_teachers"]["official_sam"][
        "persistence"
    ] = "runtime_sidecar"
    payload["source_sam_field_upgrade"][
        "source_rgb_or_official_sam_at_evaluation"
    ] = True
    candidate = tmp_path / "registry-v2.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    issues = validate_registry(candidate, root=ROOT)
    assert any("non-persistent scale-relative no-harm teacher" in issue for issue in issues)
    assert any("RGB-free evaluation" in issue for issue in issues)


def test_v2_promotion_cannot_skip_six_task_no_regression(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY_V2.read_text(encoding="utf-8"))
    payload["promotion_policy"]["require_six_task_no_regression_before_promotion"] = False
    candidate = tmp_path / "registry-v2.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    issues = validate_registry(candidate, root=ROOT)
    assert any("require_six_task_no_regression_before_promotion=true" in issue for issue in issues)
