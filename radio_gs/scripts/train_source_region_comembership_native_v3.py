#!/usr/bin/env python3
"""Train the independent native-relation candidate on frozen exact4+2."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import factorized_native_region_relation as native_interface
from radio_gs.models.region_comembership_native_v3 import (
    PAIR_FEATURE_NAMES,
    RegionCoMembershipNativeV3,
)
from radio_gs.scripts import train_source_region_comembership_v2 as v2
from radio_gs.scripts.materialize_source_region_comembership_native_v3 import (
    SCHEMA as SOURCE_AUTHORITY_SCHEMA,
    source_access as materializer_source_access,
    validate_source_region_comembership_native_v3,
)
from radio_gs.scripts.materialize_source_region_comembership_v2 import (
    densify_primitive_instance_mass,
)
from radio_gs.scripts.train_source_region_comembership_v1 import (
    LEARNING_RATE,
    SEED,
    TRAIN_SCENES,
    VALIDATION_SCENES,
    WEIGHT_DECAY,
    balanced_scene_loss,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.source_region_comembership_native_v3_execution_authority.v1"
)
CHECKPOINT_SCHEMA = "radio_gs.region_comembership_native_v3_checkpoint.v1"
RESULT_SCHEMA = "radio_gs.region_comembership_native_v3_source_result.v1"
PREREGISTRATION = Path(
    "paper/artifacts/source_only_region_comembership_native_v3_preregistration_20260807.json"
)
PARENT_V2_SOURCE_RESULT = Path(
    "paper/artifacts/source_only_region_comembership_v2_formal_result_20260807.json"
)
EPOCHS = v2.EPOCHS
SNAPSHOT_EPOCHS = v2.SNAPSHOT_EPOCHS
CALIBRATION_EPSILON = 1e-12


def source_access() -> dict[str, bool]:
    return {
        **materializer_source_access(),
        "source_validation_used_for_selection": True,
        "absolute_pair_probability_calibration_computed": True,
        "target_execution_authorized": False,
    }


def training_contract() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1,
        "model": {
            "class": "RegionCoMembershipNativeV3",
            "pair_features": list(PAIR_FEATURE_NAMES),
            "architecture": f"{len(PAIR_FEATURE_NAMES)}_64_32_1_GELU",
            "epoch_zero_probability": 0.5,
            "query_input": False,
            "scene_embedding": False,
            "legacy_v2_default_changed": False,
        },
        "cohort": {
            "source_train": list(TRAIN_SCENES),
            "source_validation": list(VALIDATION_SCENES),
            "identical_to_formal_v2": True,
        },
        "normalization": "source_train_positive_evidence_median_and_1.4826_MAD",
        "objective": "equal_scene_equal_class_evidence_weighted_BCE",
        "optimizer": {
            "name": "AdamW",
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "early_stopping": False,
        },
        "selection": {
            **v2._metric_contract(),
            "absolute_calibration": {
                "weights": "equal_class_then_pair_evidence",
                "metrics": ["brier", "log_loss", "ece10"],
                "macro_strict_improvement_over_epoch_zero": [
                    "brier",
                    "log_loss",
                ],
                "every_validation_scene_brier_non_regression": True,
            },
        },
        "preregistration": file_record(root / PREREGISTRATION),
        "parent_v2_source_result": file_record(root / PARENT_V2_SOURCE_RESULT),
        "native_interface_contract_sha256": (
            native_interface.INTERFACE_CONTRACT_SHA256
        ),
        "source_access": source_access(),
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    return v2._record(value, label=label)


def validate_execution_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("native V3 execution authority must be a mapping")
    authority = dict(value)
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "model_implementation",
        "source_materializer_implementation",
        "native_interface_implementation",
        "preregistration",
        "parent_v2_source_result",
        "source_train",
        "source_validation",
        "training_authorized",
        "target_execution_authorized",
        "source_access",
    }
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_source_only_native_v3_exact4train_2validation"
        or authority.get("training_authorized") is not True
        or authority.get("target_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("native V3 execution authority header differs")
    for name in (
        "implementation",
        "model_implementation",
        "source_materializer_implementation",
        "native_interface_implementation",
        "preregistration",
        "parent_v2_source_result",
    ):
        authority[name] = _record(authority[name], label=name)
    for split, expected in (
        ("source_train", TRAIN_SCENES),
        ("source_validation", VALIDATION_SCENES),
    ):
        rows = authority[split]
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError(f"native V3 {split} count differs")
        normalized = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"scene_id", "authority"}:
                raise ValueError(f"native V3 {split} record differs")
            normalized.append(
                {
                    "scene_id": str(row["scene_id"]),
                    "authority": _record(row["authority"], label=f"{split} scene"),
                }
            )
        if tuple(row["scene_id"] for row in normalized) != tuple(expected):
            raise ValueError(f"native V3 {split} cohort differs")
        authority[split] = normalized
    return authority


def load_scene_authority_native_v3(
    record: Mapping[str, str], *, expected_scene_id: str, expected_split: str
) -> v2.SceneAuthorityV2:
    source = validate_file_record(record, label="source native V3 scene authority")
    payload, digest, verified = load_torch_mapping(
        source,
        expected_sha256=record["sha256"],
        map_location="cpu",
        label="source native V3 scene authority",
    )
    value = validate_source_region_comembership_native_v3(payload)
    if (
        value["schema"] != SOURCE_AUTHORITY_SCHEMA
        or value["scene_id"] != expected_scene_id
        or value["split"] != expected_split
    ):
        raise ValueError("source native V3 scene identity differs")
    dense = densify_primitive_instance_mass(
        flat_keys=value["primitive_instance_flat_keys"],
        mass=value["primitive_instance_mass"],
        primitive_count=value["primitive_count"],
        instance_columns_including_zero=value[
            "instance_columns_including_zero"
        ],
    )
    core, region_mass, dominant, dominant_mass, eligible = v2._scene_core_evidence(
        value["region_rows"], value["token_mask"], dense
    )
    return v2.SceneAuthorityV2(
        scene_id=expected_scene_id,
        split=expected_split,
        record={"path": str(verified), "sha256": digest},
        region_count=int(value["canonical_region_indices"].numel()),
        pair_indices=value["pair_indices"].long().cpu().contiguous(),
        pair_features=value["pair_features"].float().cpu().contiguous(),
        targets=value["same_instance_targets"].bool().cpu().contiguous(),
        evidence_weights=value["pair_evidence_weights"].float().cpu().contiguous(),
        region_rows=value["region_rows"].long().cpu().contiguous(),
        token_mask=value["token_mask"].bool().cpu().contiguous(),
        primitive_instance_mass=dense,
        core_rows=core,
        region_instance_mass=region_mass,
        dominant_instance_ids=dominant,
        dominant_instance_mass=dominant_mass.float(),
        eligible_seeds=eligible.bool(),
        target_total=dense[:, 1:].double().sum(dim=0),
        scene_total=float(dense.double().sum()),
    )


def balanced_probability_calibration(
    probability: torch.Tensor,
    target: torch.Tensor,
    evidence_weight: torch.Tensor,
    *,
    bins: int = 10,
) -> dict[str, float]:
    """Return class-balanced absolute probability calibration metrics."""

    p = torch.as_tensor(probability).detach().double().cpu()
    y = torch.as_tensor(target).detach().bool().cpu()
    evidence = torch.as_tensor(evidence_weight).detach().double().cpu()
    if (
        p.ndim != 1
        or y.shape != p.shape
        or evidence.shape != p.shape
        or p.numel() <= 0
        or not bool(torch.isfinite(p).all())
        or not bool(torch.isfinite(evidence).all())
        or bool((p < 0).any())
        or bool((p > 1).any())
        or bool((evidence < 0).any())
        or int(bins) <= 1
    ):
        raise ValueError("native V3 calibration inputs differ")
    weights = torch.zeros_like(evidence)
    for label in (False, True):
        selected = (y == label) & (evidence > 0)
        mass = evidence[selected].sum()
        if not bool(selected.any()) or float(mass) <= 0:
            raise ValueError("native V3 calibration requires both evidence classes")
        weights[selected] = 0.5 * evidence[selected] / mass
    brier = float((weights * (p - y.double()).square()).sum())
    clipped = p.clamp(1e-7, 1.0 - 1e-7)
    log_loss = float(
        -(
            weights
            * (y.double() * clipped.log() + (~y).double() * (1.0 - clipped).log())
        ).sum()
    )
    ece = 0.0
    for index in range(int(bins)):
        lower = index / int(bins)
        upper = (index + 1) / int(bins)
        selected = (p >= lower) & (p < upper if index + 1 < bins else p <= upper)
        mass = float(weights[selected].sum())
        if mass <= 0:
            continue
        confidence = float((weights[selected] * p[selected]).sum()) / mass
        accuracy = float((weights[selected] * y[selected].double()).sum()) / mass
        ece += mass * abs(confidence - accuracy)
    return {"brier": brier, "log_loss": log_loss, "ece10": ece}


def heldout_calibration(
    scenes: Sequence[v2.SceneAuthorityV2],
    probabilities: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    per_scene = {
        scene.scene_id: balanced_probability_calibration(
            probabilities[scene.scene_id], scene.targets, scene.evidence_weights
        )
        for scene in scenes
    }
    names = ("brier", "log_loss", "ece10")
    return {
        "scene_macro": {
            name: sum(row[name] for row in per_scene.values()) / len(per_scene)
            for name in names
        },
        "per_scene": per_scene,
    }


def calibration_gate(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, bool]:
    selected = candidate["pair_calibration"]
    epoch_zero = baseline["pair_calibration"]
    per_scene_non_regression = all(
        selected["per_scene"][scene]["brier"]
        <= epoch_zero["per_scene"][scene]["brier"] + CALIBRATION_EPSILON
        for scene in VALIDATION_SCENES
    )
    return {
        "macro_brier_strictly_improves_epoch_zero": (
            selected["scene_macro"]["brier"]
            < epoch_zero["scene_macro"]["brier"] - CALIBRATION_EPSILON
        ),
        "macro_log_loss_strictly_improves_epoch_zero": (
            selected["scene_macro"]["log_loss"]
            < epoch_zero["scene_macro"]["log_loss"] - CALIBRATION_EPSILON
        ),
        "every_validation_scene_brier_non_regression": per_scene_non_regression,
    }


def _parse_authority_records(values: Sequence[str]) -> dict[str, dict[str, str]]:
    return v2._parse_authority_records(values)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"native V3 execution authority exists: {output}")
    records = _parse_authority_records(args.scene_authority)
    for scene in (*TRAIN_SCENES, *VALIDATION_SCENES):
        load_scene_authority_native_v3(
            records[scene],
            expected_scene_id=scene,
            expected_split=(
                "source_train" if scene in TRAIN_SCENES else "source_validation"
            ),
        )
    root = Path(__file__).resolve().parents[2]
    authority = {
        "schema": EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_native_v3_exact4train_2validation",
        "implementation": file_record(Path(__file__).resolve()),
        "model_implementation": file_record(
            root / "radio_gs/models/region_comembership_native_v3.py"
        ),
        "source_materializer_implementation": file_record(
            root
            / "radio_gs/scripts/materialize_source_region_comembership_native_v3.py"
        ),
        "native_interface_implementation": file_record(
            Path(native_interface.__file__).resolve()
        ),
        "preregistration": file_record(root / PREREGISTRATION),
        "parent_v2_source_result": file_record(root / PARENT_V2_SOURCE_RESULT),
        "source_train": [
            {"scene_id": scene, "authority": records[scene]}
            for scene in TRAIN_SCENES
        ],
        "source_validation": [
            {"scene_id": scene, "authority": records[scene]}
            for scene in VALIDATION_SCENES
        ],
        "training_authorized": True,
        "target_execution_authorized": False,
        "source_access": source_access(),
    }
    validate_execution_authority(authority)
    write_frozen_json(output, authority)
    return {
        "status": "native_V3_execution_authority_complete",
        "output": file_record(output),
    }


def _load_execution(args: argparse.Namespace) -> tuple[
    dict[str, Any], tuple[v2.SceneAuthorityV2, ...], tuple[v2.SceneAuthorityV2, ...]
]:
    raw, digest, source = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="native V3 execution authority",
    )
    authority = validate_execution_authority(raw)
    root = Path(__file__).resolve().parents[2]
    expected_paths = {
        "implementation": Path(__file__).resolve(),
        "model_implementation": (
            root / "radio_gs/models/region_comembership_native_v3.py"
        ),
        "source_materializer_implementation": (
            root
            / "radio_gs/scripts/materialize_source_region_comembership_native_v3.py"
        ),
        "native_interface_implementation": Path(native_interface.__file__).resolve(),
        "preregistration": root / PREREGISTRATION,
        "parent_v2_source_result": root / PARENT_V2_SOURCE_RESULT,
    }
    for name, expected in expected_paths.items():
        if validate_file_record(authority[name], label=f"native V3 {name}") != expected.resolve():
            raise ValueError(f"native V3 authority binds another {name}")

    def load(split: str, expected: Sequence[str]) -> tuple[v2.SceneAuthorityV2, ...]:
        return tuple(
            load_scene_authority_native_v3(
                row["authority"], expected_scene_id=scene, expected_split=split
            )
            for row, scene in zip(authority[split], expected)
        )

    authority["verified_path"] = str(source)
    authority["verified_sha256"] = digest
    return (
        authority,
        load("source_train", TRAIN_SCENES),
        load("source_validation", VALIDATION_SCENES),
    )


def _state_copy(model: RegionCoMembershipNativeV3) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_json_sha256(
        {name: tensor_sha256(value) for name, value in sorted(state.items())}
    )


def validate_checkpoint(value: object) -> dict[str, Any]:
    """Validate an independent source-only native-V3 candidate checkpoint."""

    if not isinstance(value, Mapping):
        raise ValueError("native V3 checkpoint must be a mapping")
    checkpoint = dict(value)
    required = {
        "schema",
        "schema_version",
        "training_contract",
        "training_contract_sha256",
        "execution_authority",
        "feature_names",
        "normalization",
        "model_state_dict",
        "model_state_dict_sha256",
        "selected_epoch",
        "selected_rule",
        "selected_validation",
        "singleton_validation",
        "promotion_gate",
        "source_access",
        "target_execution_performed",
    }
    contract = checkpoint.get("training_contract")
    normalization = checkpoint.get("normalization")
    state = checkpoint.get("model_state_dict")
    selected = checkpoint.get("selected_validation")
    singleton = checkpoint.get("singleton_validation")
    rule = checkpoint.get("selected_rule")
    gate = checkpoint.get("promotion_gate")
    if (
        set(checkpoint) != required
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("schema_version") != 1
        or contract != training_contract()
        or checkpoint.get("training_contract_sha256")
        != canonical_json_sha256(contract)
        or checkpoint.get("feature_names") != list(PAIR_FEATURE_NAMES)
        or checkpoint.get("source_access") != source_access()
        or checkpoint.get("target_execution_performed") is not False
        or not isinstance(normalization, Mapping)
        or set(normalization) != {"median", "robust_scale"}
        or not isinstance(state, Mapping)
        or checkpoint.get("model_state_dict_sha256") != _state_sha(state)
        or int(checkpoint.get("selected_epoch", -1)) not in SNAPSHOT_EPOCHS
        or not isinstance(rule, Mapping)
        or set(rule) != {"method", "maximum_regions", "threshold"}
        or rule.get("method") not in v2.METHODS
        or int(rule.get("maximum_regions", 0)) not in v2.MAXIMUM_REGIONS
        or float(rule.get("threshold", -1.0)) not in v2.THRESHOLDS
        or not isinstance(selected, Mapping)
        or not isinstance(singleton, Mapping)
        or int(selected.get("epoch", -1)) != int(checkpoint["selected_epoch"])
        or any(selected.get(name) != rule[name] for name in rule)
        or int(singleton.get("epoch", -1)) != 0
        or int(singleton.get("maximum_regions", 0)) != 1
        or not isinstance(gate, Mapping)
    ):
        raise ValueError("native V3 checkpoint contract differs")
    dimension = len(PAIR_FEATURE_NAMES)
    median = torch.as_tensor(normalization["median"])
    scale = torch.as_tensor(normalization["robust_scale"])
    if (
        median.dtype != torch.float32
        or scale.dtype != torch.float32
        or median.shape != (dimension,)
        or scale.shape != (dimension,)
        or not bool(torch.isfinite(median).all())
        or not bool(torch.isfinite(scale).all())
        or bool((scale <= 0).any())
    ):
        raise ValueError("native V3 checkpoint normalization differs")
    model = RegionCoMembershipNativeV3(median, scale)
    try:
        model.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("native V3 checkpoint model state differs") from error
    _record(checkpoint["execution_authority"], label="checkpoint execution authority")
    topology_names = (
        "selected_epoch_positive",
        "topology_strictly_exceeds_singleton",
        "iou_strictly_exceeds_singleton",
        "f1_strictly_exceeds_singleton",
    )
    calibration_names = (
        "macro_brier_strictly_improves_epoch_zero",
        "macro_log_loss_strictly_improves_epoch_zero",
        "every_validation_scene_brier_non_regression",
    )
    if any(name not in gate or not isinstance(gate[name], bool) for name in (*topology_names, *calibration_names)):
        raise ValueError("native V3 checkpoint promotion gate differs")
    expected_passed = all(bool(gate[name]) for name in (*topology_names, *calibration_names))
    if gate.get("passed") is not expected_passed:
        raise ValueError("native V3 checkpoint promotion verdict differs")
    return checkpoint


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("native V3 checkpoint/report output must be new")
    execution, train_scenes, validation_scenes = _load_execution(args)
    median, scale = v2.fit_normalization(train_scenes)
    device = torch.device(str(args.device))
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = RegionCoMembershipNativeV3(median, scale).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    device_rows = {
        scene.scene_id: (
            scene.pair_features.to(device),
            scene.targets.to(device),
            scene.evidence_weights.to(device),
        )
        for scene in train_scenes
    }
    snapshots = {0: _state_copy(model)}
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for scene in train_scenes:
            features, targets, weights = device_rows[scene.scene_id]
            loss = balanced_scene_loss(model(features), targets, weights)
            (loss / len(train_scenes)).backward()
            losses.append(float(loss.detach().cpu()))
        optimizer.step()
        row = {
            "epoch": epoch,
            "train_scene_macro_balanced_bce": sum(losses) / len(losses),
        }
        history.append(row)
        if epoch in SNAPSHOT_EPOCHS:
            snapshots[epoch] = _state_copy(model)
            print(json.dumps(row, sort_keys=True), flush=True)
    if tuple(sorted(snapshots)) != SNAPSHOT_EPOCHS:
        raise RuntimeError("native V3 retained checkpoint schedule differs")
    model.cpu()
    del device_rows
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    exact_candidates = []
    proxy_audit = {}
    calibration_by_epoch = {}
    for epoch in SNAPSHOT_EPOCHS:
        model.load_state_dict(snapshots[epoch], strict=True)
        model.eval()
        with torch.no_grad():
            probabilities = {
                scene.scene_id: model.probability(scene.pair_features).cpu()
                for scene in validation_scenes
            }
        calibration = heldout_calibration(validation_scenes, probabilities)
        calibration_by_epoch[str(epoch)] = calibration
        proxy = v2._proxy_grid(validation_scenes, probabilities)
        shortlist = v2.deterministic_proxy_shortlist(proxy)
        proxy_audit[str(epoch)] = {
            "grid_rows": len(proxy),
            "pair_calibration": calibration,
            "shortlist": [
                {
                    "method": row["method"],
                    "maximum_regions": row["maximum_regions"],
                    "threshold": row["threshold"],
                    "scene_macro": row["scene_macro"],
                }
                for row in shortlist
            ],
        }
        exact_selection_cache: dict[
            tuple[str, str, float], dict[int, tuple[int, ...]]
        ] = {}
        for rule in shortlist:
            candidate = v2._exact_candidate(
                epoch=epoch,
                rule=rule,
                scenes=validation_scenes,
                probabilities=probabilities,
                selection_cache=exact_selection_cache,
            )
            candidate["pair_calibration"] = calibration
            exact_candidates.append(candidate)

    baseline = next(
        row
        for row in exact_candidates
        if row["epoch"] == 0 and row["maximum_regions"] == 1
    )
    eligible = [
        row
        for row in exact_candidates
        if int(row["epoch"]) > 0
        and all(calibration_gate(row, baseline).values())
    ]
    selection_pool = eligible if eligible else exact_candidates
    selected = max(
        selection_pool, key=lambda row: v2._rule_key(row, include_epoch=True)
    )
    topology = {
        "selected_epoch_positive": int(selected["epoch"]) > 0,
        "topology_strictly_exceeds_singleton": (
            selected["scene_macro"]["topology_score"]
            > baseline["scene_macro"]["topology_score"]
        ),
        "iou_strictly_exceeds_singleton": (
            selected["scene_macro"]["iou"] > baseline["scene_macro"]["iou"]
        ),
        "f1_strictly_exceeds_singleton": (
            selected["scene_macro"]["f1"] > baseline["scene_macro"]["f1"]
        ),
    }
    calibration = calibration_gate(selected, baseline)
    promotion = {
        **topology,
        **calibration,
        "singleton": baseline["scene_macro"],
        "selected": selected["scene_macro"],
        "epoch_zero_pair_calibration": baseline["pair_calibration"],
        "selected_pair_calibration": selected["pair_calibration"],
        "calibration_eligible_candidate_count": len(eligible),
    }
    promotion["passed"] = all(
        promotion[name]
        for name in (
            *topology,
            *calibration,
        )
    )
    selected_state = snapshots[int(selected["epoch"])]
    contract = training_contract()
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "training_contract": contract,
        "training_contract_sha256": canonical_json_sha256(contract),
        "execution_authority": {
            "path": execution["verified_path"],
            "sha256": execution["verified_sha256"],
        },
        "feature_names": list(PAIR_FEATURE_NAMES),
        "normalization": {"median": median, "robust_scale": scale},
        "model_state_dict": selected_state,
        "model_state_dict_sha256": _state_sha(selected_state),
        "selected_epoch": int(selected["epoch"]),
        "selected_rule": {
            name: selected[name]
            for name in ("method", "maximum_regions", "threshold")
        },
        "selected_validation": selected,
        "singleton_validation": baseline,
        "promotion_gate": promotion,
        "source_access": source_access(),
        "target_execution_performed": False,
    }
    validate_checkpoint(checkpoint)
    checkpoint_path = write_torch_noclobber(output, checkpoint)
    report = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": "source_only_native_v3_exact4train_2validation_complete",
        "checkpoint": file_record(checkpoint_path),
        "selected_validation": selected,
        "singleton_validation": baseline,
        "promotion_gate": promotion,
        "exact_candidate_count": len(exact_candidates),
        "exact_candidates": exact_candidates,
        "calibration_by_epoch": calibration_by_epoch,
        "proxy_audit": proxy_audit,
        "history": history,
        "source_access": source_access(),
        "target_execution_performed": False,
    }
    write_frozen_json(report_path, report)
    return report


def synthetic_dry_run() -> dict[str, Any]:
    torch.manual_seed(SEED)
    dimension = len(PAIR_FEATURE_NAMES)
    model = RegionCoMembershipNativeV3(torch.zeros(dimension), torch.ones(dimension))
    features = torch.randn(8, dimension)
    target = torch.tensor([True, False] * 4)
    weight = torch.ones(8)
    logits = model(features)
    loss = balanced_scene_loss(logits, target, weight)
    loss.backward()
    calibration = balanced_probability_calibration(
        torch.sigmoid(logits), target, weight
    )
    return {
        "feature_dimension": dimension,
        "epoch_zero_probability": float(torch.sigmoid(logits).mean()),
        "final_layer_gradient_nonzero": bool(
            model.network[-1].weight.grad.abs().sum() > 0
        ),
        "epoch_zero_calibration": calibration,
        "snapshot_epochs": list(SNAPSHOT_EPOCHS),
        "legacy_v2_default_changed": False,
        "benchmark_opened": False,
        "target_metric_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic-dry-run")
    prepare = commands.add_parser("preflight")
    prepare.add_argument("--scene-authority", action="append", required=True)
    prepare.add_argument("--output", required=True)
    run = commands.add_parser("train")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--expected-execution-authority-sha256", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "synthetic-dry-run":
        result = synthetic_dry_run()
    elif args.command == "preflight":
        result = preflight(args)
    else:
        result = train(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
