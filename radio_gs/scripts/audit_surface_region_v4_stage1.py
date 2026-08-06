#!/usr/bin/env python3
"""Strict, query-free stage-1 audit for a SurfaceRegion V4 cache.

This audit deliberately does not reconstruct the scene graph.  It proves only
claims represented by the immutable cache and its SHA-bound replay authority.
In particular, a declared ``token_candidate_limit`` is not evidence that the
bounded graph search was complete; the report enumerates the rows that require
graph replay before making such a claim.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_contract import (
    SurfaceRegionContractV4,
)
from radio_gs.interfaces.surface_region_selection import (
    surface_region_contract_from_specification,
)
from radio_gs.training.surface_region_eligibility_completion import (
    STRUCTURED_ELIGIBILITY_POLICY,
)
from radio_gs.scripts.surface_region_scene_resume import SCENE_ROW_SCHEMA_V3
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    write_frozen_json,
)


_TEACHER_KEYS = (
    "official_summary_tokens",
    "official_crop_summaries",
    "teacher_mask",
)
_TEACHER_IDENTITY_KEYS = (
    "scene",
    "seed",
    "physical_radius_m",
    "teacher_views",
    "teacher_medoid",
    "teacher_region_tokens",
    "teacher_support_sha256",
    "teacher_target_sha256",
)


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _tensor(payload: dict[str, Any], key: str) -> torch.Tensor:
    _require(key in payload, f"V4 cache lacks {key}")
    return torch.as_tensor(payload[key]).cpu()


def _distribution(values: torch.Tensor) -> dict[str, int | float | None]:
    flat = torch.as_tensor(values).detach().double().reshape(-1)
    if flat.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "minimum": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "maximum": None,
        }
    _require(bool(torch.isfinite(flat).all()), "risk proxy contains non-finite values")
    quantiles = torch.quantile(
        flat, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.double)
    )
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "minimum": float(flat.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "maximum": float(flat.max()),
    }


class _CosineHistogram:
    """Exact counts/moments and a fixed-bin pairwise cosine distribution."""

    def __init__(self) -> None:
        self.edges = torch.linspace(-1.0, 1.0, 41, dtype=torch.double)
        self.counts = torch.zeros(40, dtype=torch.long)
        self.count = 0
        self.total = 0.0
        self.total_squared = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, values: torch.Tensor) -> None:
        flat = torch.as_tensor(values).detach().double().reshape(-1)
        if not flat.numel():
            return
        _require(bool(torch.isfinite(flat).all()), "cosine proxy is non-finite")
        flat = flat.clamp(-1.0, 1.0)
        self.count += int(flat.numel())
        self.total += float(flat.sum())
        self.total_squared += float((flat * flat).sum())
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))
        # bucketize against the 39 interior boundaries; both endpoints remain
        # in the first/last of the forty closed outer bins.
        indices = torch.bucketize(flat, self.edges[1:-1], right=False)
        self.counts += torch.bincount(indices, minlength=40)

    def report(self) -> dict[str, Any]:
        mean = self.total / self.count if self.count else None
        variance = (
            max(0.0, self.total_squared / self.count - float(mean) ** 2)
            if self.count
            else None
        )
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance) if variance is not None else None,
            "minimum": self.minimum if self.count else None,
            "maximum": self.maximum if self.count else None,
            "bin_edges": [float(value) for value in self.edges],
            "bin_counts": [int(value) for value in self.counts],
        }


def _cat(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(values) if values else torch.empty(0, dtype=torch.float32)


def _record_roles(records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, list[int]]]:
    full: dict[str, int] = {}
    completion: dict[str, list[int]] = {}
    seen: set[str] = set()
    for row, record in enumerate(records):
        role = record.get("row_role")
        region_id = str(record.get("region_id", ""))
        paired = str(record.get("paired_full_region_id", ""))
        _require(bool(region_id and paired), f"row {row} lacks paired identity")
        _require(region_id not in seen, f"row {row} duplicates region identity")
        seen.add(region_id)
        if role == "full_support":
            _require(paired == region_id and region_id not in full, f"row {row} full identity differs")
            _require(int(record.get("eligibility_variant_index", -2)) == -1, f"row {row} full variant differs")
            full[region_id] = row
        elif role == "eligibility_completion":
            digest = str(record.get("eligibility_sha256", ""))
            _require(
                len(digest) == 64
                and all(value in "0123456789abcdef" for value in digest),
                f"row {row} eligibility digest differs",
            )
            _require(
                record.get("eligibility_policy") == STRUCTURED_ELIGIBILITY_POLICY,
                f"row {row} eligibility policy differs",
            )
            completion.setdefault(paired, []).append(row)
        else:
            raise ValueError(f"row {row} has unknown role")
    _require(set(completion) == set(full), "full/completion pair coverage differs")
    return full, completion


def _validate_contract(metadata: dict[str, Any]) -> SurfaceRegionContractV4:
    raw = metadata.get("region_contract")
    _require(isinstance(raw, dict), "V4 cache lacks its region contract")
    contract = surface_region_contract_from_specification(raw)
    _require(type(contract) is SurfaceRegionContractV4, "cache does not use exact contract V4")
    _require(metadata.get("region_contract_version") == contract.version, "V4 contract version differs")
    _require(metadata.get("region_contract_sha256") == contract.digest, "V4 contract digest differs")
    _require(metadata.get("schema_version") == 4, "V4 cache schema differs")
    _require(
        metadata.get("surface_region_row_schema_version") == SCENE_ROW_SCHEMA_V3,
        "V4 row schema differs",
    )
    _require(metadata.get("training_scope") == "global_cross_scene_3d_surface_v4", "V4 training scope differs")
    return contract


def _validate_replay_binding(
    metadata: dict[str, Any],
    source_metadata: dict[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
) -> None:
    replay = metadata.get("teacher_replay_cache")
    _require(isinstance(replay, dict), "V4 cache lacks replay provenance")
    _require(replay.get("sha256") == source_sha256, "embedded replay SHA-256 differs")
    _require(Path(str(replay.get("path", ""))).resolve() == source_path, "embedded replay path differs")
    _require(metadata.get("teacher_target_source") == "exact_cache_replay", "V4 teacher source is not exact replay")
    _require(source_metadata.get("teacher_target_source") == "fresh_official_runtime", "replay authority is not fresh")
    _require(source_metadata.get("complete_scene_regions") is True, "replay authority is incomplete")
    _require(not source_metadata.get("failed_scenes"), "replay authority contains failed scenes")
    _require(source_metadata.get("teacher_regions_saturated") == 0, "replay authority has saturated teacher regions")
    for key in (
        "teacher_region_contract_sha256",
        "teacher_target_protocol_sha256",
        "radio_checkpoint_sha256",
        "split_role",
        "split_file_sha256",
    ):
        _require(metadata.get(key) == source_metadata.get(key), f"replay authority {key} differs")


def _validate_teacher_replay(
    cache: dict[str, Any],
    source: dict[str, Any],
    records: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    full: dict[str, int],
    completion: dict[str, list[int]],
) -> dict[str, Any]:
    cache_tensors = {key: _tensor(cache, key) for key in _TEACHER_KEYS}
    source_tensors = {key: torch.as_tensor(source[key]).cpu() for key in _TEACHER_KEYS}
    source_full_rows = {
        row
        for row, record in enumerate(source_records)
        if record.get("row_role", "full_support") == "full_support"
    }
    source_paired_rows = 0
    if source.get("metadata", {}).get("schema_version") == 4:
        source_full, source_completion = _record_roles(source_records)
        source_full_rows = set(source_full.values())
        for source_full_id, source_full_row in source_full.items():
            for row in source_completion[source_full_id]:
                source_paired_rows += 1
                for key in _TEACHER_IDENTITY_KEYS:
                    _require(
                        source_records[row].get(key)
                        == source_records[source_full_row].get(key),
                        f"source row {row} teacher identity {key} differs",
                    )
                for key in _TEACHER_KEYS:
                    _require(
                        torch.equal(
                            source_tensors[key][row],
                            source_tensors[key][source_full_row],
                        ),
                        f"source pair {source_full_id} teacher tensor {key} differs",
                    )
    mapped_full_rows: set[int] = set()
    for full_id, full_row in full.items():
        paired_rows = completion[full_id]
        variants = int(records[full_row].get("eligibility_variants_per_teacher_region", 0))
        _require(variants > 0 and len(paired_rows) == variants, f"pair {full_id} variant coverage differs")
        _require(
            sorted(int(records[row].get("eligibility_variant_index", -1)) for row in paired_rows)
            == list(range(variants)),
            f"pair {full_id} variant indices differ",
        )
        source_row = int(records[full_row].get("teacher_replay_source_row", -1))
        _require(source_row in source_full_rows, f"pair {full_id} maps outside source full rows")
        _require(source_row not in mapped_full_rows, f"source row {source_row} is mapped twice")
        mapped_full_rows.add(source_row)
        source_record = source_records[source_row]
        _require(
            records[full_row].get("teacher_replay_source_region_id")
            == source_record.get("region_id"),
            f"pair {full_id} source region identity differs",
        )
        for row in [full_row, *paired_rows]:
            _require(int(records[row].get("teacher_replay_source_row", -1)) == source_row, f"row {row} source row differs")
            for key in _TEACHER_IDENTITY_KEYS:
                _require(records[row].get(key) == source_record.get(key), f"row {row} teacher identity {key} differs")
            for key in _TEACHER_KEYS:
                _require(torch.equal(cache_tensors[key][row], source_tensors[key][source_row]), f"row {row} teacher tensor {key} differs bitwise")
                _require(torch.equal(cache_tensors[key][row], cache_tensors[key][full_row]), f"pair {full_id} teacher tensor {key} differs")
    _require(mapped_full_rows == source_full_rows, "V4 replay does not bijectively cover source full rows")
    teacher_mask = cache_tensors["teacher_mask"].bool()
    _require(bool(teacher_mask.any(dim=1).all()), "teacher mask has an empty row")
    for key in ("official_summary_tokens", "official_crop_summaries"):
        values = cache_tensors[key]
        _require(bool((values.masked_select(~teacher_mask[..., None])).eq(0).all()), f"{key} teacher padding is nonzero")
    return {
        "status": "bitwise_exact_bijective_full_row_replay_verified",
        "source_full_rows": len(source_full_rows),
        "source_completion_rows_verified": source_paired_rows,
        "v4_full_rows": len(full),
        "v4_completion_rows": sum(len(value) for value in completion.values()),
        "tensor_keys": list(_TEACHER_KEYS),
        "identity_keys": list(_TEACHER_IDENTITY_KEYS),
    }


def _validate_rows_and_measure(
    cache: dict[str, Any],
    records: list[dict[str, Any]],
    contract: SurfaceRegionContractV4,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    features = _tensor(cache, "radio_features")
    geometry = _tensor(cache, "geometry")
    mask = _tensor(cache, "token_mask").bool()
    reliability = _tensor(cache, "reliability")
    fill = _tensor(cache, "support_fill_mask").bool()
    anchors = _tensor(cache, "anchor_index").long()
    rows, tokens = mask.shape
    _require(features.shape == (rows, tokens, 1280), "radio feature shape differs")
    _require(geometry.shape == (rows, tokens, 16), "geometry shape differs")
    _require(reliability.shape == (rows, tokens, 1), "reliability shape differs")
    _require(anchors.shape == (rows,), "anchor index shape differs")
    _require(tokens == contract.maximum_tokens and rows == len(records), "row/token extent differs")
    for value, label in ((features, "radio_features"), (geometry, "geometry"), (reliability, "reliability")):
        _require(bool(torch.isfinite(value).all()), f"{label} contains non-finite values")
        _require(bool(value.masked_select(~mask[..., None]).eq(0).all()), f"{label} padding is nonzero")
    _require(not bool((fill & ~mask).any()), "support-fill marks padding")

    flags = geometry[..., [7, 8, 9, 14]].float()
    _require(bool(((flags == 0) | (flags == 1)).all()), "geometry membership flags are not binary")
    anchor_flag, core, context, geometry_fill = [flags[..., index].bool() for index in range(4)]
    _require(torch.equal(fill, geometry_fill), "support-fill mask and geometry index 14 differ")
    _require(torch.equal(reliability, geometry[..., 6:7]), "reliability and authoritative geometry index 6 differ")
    memberships = core.to(torch.int8) + context.to(torch.int8) + fill.to(torch.int8)
    _require(bool((memberships[mask] == 1).all()) and not bool(memberships[~mask].any()), "core/context/fill masks do not partition active tokens")
    _require(bool(anchor_flag.sum(1).eq(1).all()), "each row must have exactly one anchor")
    _require(bool((anchors >= 0).all()) and bool((anchors < tokens).all()), "anchor index is outside row")
    batch = torch.arange(rows)
    _require(bool(anchor_flag[batch, anchors].all()) and bool(core[batch, anchors].all()), "anchor flag/index/core binding differs")

    selected_norm = torch.linalg.vector_norm(features.float(), dim=-1)[mask]
    _require(
        torch.allclose(selected_norm, torch.ones_like(selected_norm), rtol=2e-3, atol=2e-3),
        "active RADIO directions are not unit norm within fp16 storage tolerance",
    )

    roles = {"core": core, "context": context, "support_fill": fill}
    by_role: dict[str, dict[str, list[torch.Tensor]]] = {
        role: {name: [] for name in ("reliability", "log_raw_norm", "raw_norm", "relative_distance", "relative_scale")}
        for role in roles
    }
    anchor_context: list[torch.Tensor] = []
    nearest_core: list[torch.Tensor] = []
    row_mean_core_context: list[torch.Tensor] = []
    raw_log_shifts: list[torch.Tensor] = []
    pairwise = _CosineHistogram()
    count_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    high_priority: list[dict[str, Any]] = []
    core_budget = max(1, min(tokens, round(tokens * contract.core_token_fraction)))
    context_budget = tokens - core_budget

    for row, record in enumerate(records):
        row_counts = {
            "core": int(core[row].sum()),
            "context": int(context[row].sum()),
            "support_fill": int(fill[row].sum()),
            "selected": int(mask[row].sum()),
        }
        for name, key in (("selected", "tokens"), ("core", "core_tokens"), ("context", "context_tokens"), ("support_fill", "support_fill_tokens")):
            _require(row_counts[name] == int(record.get(key, -1)), f"row {row} record {key} differs")
        _require(row_counts["selected"] == row_counts["core"] + row_counts["context"] + row_counts["support_fill"], f"row {row} token accounting differs")
        _require(int(record.get("semantic_tokens", -1)) == row_counts["core"] + row_counts["context"], f"row {row} semantic count differs")
        _require(record.get("minimum_satisfied") is True and contract.minimum_tokens <= row_counts["selected"] <= tokens, f"row {row} token bounds differ")
        _require(int(record.get("anchor_local_index", -1)) == int(anchors[row]), f"row {row} anchor record differs")
        count_rows.append(row_counts)

        active_geometry = geometry[row].float()
        for role, role_mask in roles.items():
            selected = role_mask[row]
            log_norm = active_geometry[selected, 15]
            values = by_role[role]
            values["reliability"].append(active_geometry[selected, 6])
            values["log_raw_norm"].append(log_norm)
            values["raw_norm"].append(torch.exp(log_norm))
            values["relative_distance"].append(torch.linalg.vector_norm(active_geometry[selected, 0:3], dim=-1))
            values["relative_scale"].append(torch.exp(active_geometry[selected, 3:6].mean(dim=-1)))

        core_features = F.normalize(features[row, core[row]].float(), dim=-1)
        context_features = F.normalize(features[row, context[row]].float(), dim=-1)
        if len(context_features):
            similarities = context_features @ core_features.T
            pairwise.add(similarities)
            nearest_core.append(similarities.max(dim=1).values)
            row_mean_core_context.append(similarities.mean().reshape(1))
            anchor_feature = F.normalize(features[row, anchors[row]].float(), dim=-1)
            anchor_context.append(context_features @ anchor_feature)
            core_log = geometry[row, core[row], 15].float().mean()
            context_log = geometry[row, context[row], 15].float().mean()
            raw_log_shifts.append((context_log - core_log).reshape(1))

        graph_record = {
            "row": row,
            "region_id": str(record["region_id"]),
            "scene": str(record["scene"]),
            "role": str(record["row_role"]),
            "seed": int(record["seed"]),
            "physical_radius_m": float(record["physical_radius_m"]),
            **row_counts,
            "reason": "cache_has_no_pre_budget_candidate_count_or_dijkstra_frontier",
        }
        graph_rows.append(graph_record)
        saturation_reasons = []
        if row_counts["selected"] == tokens:
            saturation_reasons.append("selected_token_limit_reached")
        if row_counts["core"] >= core_budget:
            saturation_reasons.append("reserved_core_budget_reached_or_donated")
        if row_counts["context"] >= context_budget and context_budget:
            saturation_reasons.append("reserved_context_budget_reached_or_donated")
        if saturation_reasons:
            high_priority.append({**graph_record, "priority_reasons": saturation_reasons})

    completion = cache["metadata"].get("eligibility_completion")
    _require(
        isinstance(completion, dict)
        and completion.get("schema_version") == 1
        and completion.get("policy") == STRUCTURED_ELIGIBILITY_POLICY
        and completion.get("validation_checkpoint_selection") == "full_support_rows_only",
        "completion selection policy differs",
    )
    variants = int(completion.get("variants_per_teacher_region", -1))
    _require(int(completion.get("full_support_rows", -1)) == sum(record["row_role"] == "full_support" for record in records), "full row aggregate differs")
    _require(int(completion.get("completion_variant_rows", -1)) == sum(record["row_role"] == "eligibility_completion" for record in records), "completion row aggregate differs")
    _require(
        variants > 0
        and all(
            int(record.get("eligibility_variants_per_teacher_region", -1))
            == variants
            for record in records
        ),
        "completion variant aggregate differs",
    )
    completion_rows = [
        (row, record)
        for row, record in enumerate(records)
        if record["row_role"] == "eligibility_completion"
    ]
    _require(
        int(completion.get("completion_rows_with_fill", -1))
        == sum(int(record["support_fill_tokens"]) > 0 for _, record in completion_rows),
        "completion fill-row aggregate differs",
    )
    _require(
        int(completion.get("completion_support_fill_tokens", -1))
        == sum(int(record["support_fill_tokens"]) for _, record in completion_rows),
        "completion fill-token aggregate differs",
    )
    _require(
        int(completion.get("completion_selected_tokens", -1))
        == sum(int(record["tokens"]) for _, record in completion_rows),
        "completion selected-token aggregate differs",
    )
    _require(int(cache["metadata"].get("semantic_tokens_total", -1)) == sum(row["core"] + row["context"] for row in count_rows), "semantic aggregate differs")
    _require(int(cache["metadata"].get("support_fill_tokens_total", -1)) == sum(row["support_fill"] for row in count_rows), "fill aggregate differs")

    risk_by_role = {
        role: {name: _distribution(_cat(values)) for name, values in measurements.items()}
        for role, measurements in by_role.items()
    }
    risk = {
        "scope": "query_free_descriptive_risk_proxies_not_semantic_purity",
        "semantic_purity_claim": False,
        "semantic_purity_limitation": (
            "RADIO cosine, reliability, geometry, and any depth-consistency signal are proxies; "
            "depth consistency measures visibility/geometric agreement and must not be relabeled semantic purity."
        ),
        "by_membership": risk_by_role,
        "core_context_pairwise_radio_cosine": pairwise.report(),
        "context_nearest_core_radio_cosine": _distribution(_cat(nearest_core)),
        "anchor_context_radio_cosine": _distribution(_cat(anchor_context)),
        "per_row_mean_core_context_radio_cosine": _distribution(_cat(row_mean_core_context)),
        "per_row_context_minus_core_mean_log_raw_norm": _distribution(_cat(raw_log_shifts)),
    }
    counts_report = {
        "rows": rows,
        "selected_tokens": sum(row["selected"] for row in count_rows),
        "core_tokens": sum(row["core"] for row in count_rows),
        "context_tokens": sum(row["context"] for row in count_rows),
        "support_fill_tokens": sum(row["support_fill"] for row in count_rows),
        "context_zero_rows": sum(row["context"] == 0 for row in count_rows),
        "core_maximum_token_rows": sum(row["core"] == tokens for row in count_rows),
        "selected_maximum_token_rows": sum(row["selected"] == tokens for row in count_rows),
        "core_reserved_budget": core_budget,
        "context_reserved_budget": context_budget,
        "core_reserved_budget_reached_rows": sum(row["core"] >= core_budget for row in count_rows),
        "context_reserved_budget_reached_rows": sum(row["context"] >= context_budget for row in count_rows) if context_budget else 0,
    }
    cap = {
        "declared_token_candidate_limit": int(contract.token_candidate_limit),
        "declared_maximum_tokens": tokens,
        "cache_proves": [
            "selected row length and typed core/context/fill counts",
            "selected rows do not exceed maximum_tokens",
            "which selected rows reach maximum_tokens or typed reserved budgets",
        ],
        "cache_does_not_prove": [
            "number of strict or soft graph candidates before typed budgeting",
            "whether bounded Dijkstra reached token_candidate_limit",
            "whether an unsettled valid candidate existed beyond the stored frontier",
            "candidate-complete equivalence to an uncapped graph traversal",
        ],
        "candidate_complete_claim_authorized": False,
        "graph_replay_required_rows": graph_rows,
        "graph_replay_required_row_count": len(graph_rows),
        "high_priority_saturated_rows": high_priority,
        "high_priority_saturated_row_count": len(high_priority),
        "required_graph_replay_evidence": [
            "reconstruct the SHA-bound scene graph and eligibility mask",
            "replay the exact anchor, radius, contract, and distance-then-node tie break",
            "record pre-budget strict/soft candidate counts and whether the 1024th candidate was settled",
            "record the next valid frontier distance/node or exhaustion of the queue",
            "compare selected typed rows against an uncapped traversal",
        ],
    }
    return counts_report, risk, cap


def audit_v4_stage1(
    cache_path: str | Path,
    *,
    cache_sha256: str,
    replay_cache_path: str | Path,
    replay_cache_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    cache, observed_cache_sha, cache_source = load_torch_mapping(
        cache_path,
        expected_sha256=cache_sha256,
        map_location="cpu",
        label="SurfaceRegion V4 cache",
    )
    source, observed_source_sha, source_path = load_torch_mapping(
        replay_cache_path,
        expected_sha256=replay_cache_sha256,
        map_location="cpu",
        label="SurfaceRegion replay cache",
    )
    metadata = cache.get("metadata")
    source_metadata = source.get("metadata")
    _require(isinstance(metadata, dict) and isinstance(source_metadata, dict), "cache metadata is missing")
    records = metadata.get("region_records")
    source_records = source_metadata.get("region_records")
    _require(isinstance(records, list) and records and all(isinstance(value, dict) for value in records), "V4 region records differ")
    _require(isinstance(source_records, list) and source_records and all(isinstance(value, dict) for value in source_records), "source region records differ")
    contract = _validate_contract(metadata)
    _validate_replay_binding(
        metadata,
        source_metadata,
        source_path=source_path,
        source_sha256=observed_source_sha,
    )
    full, completion = _record_roles(records)
    teacher = _validate_teacher_replay(cache, source, records, source_records, full, completion)
    counts, risk, cap = _validate_rows_and_measure(cache, records, contract)
    report = {
        "artifact_type": "surface-region-v4-stage1-cache-audit-v1",
        "schema_version": 1,
        "status": "passed",
        "query_free": True,
        "auditor": {
            "path": str(Path(__file__).resolve()),
            "sha256": _script_sha256(),
        },
        "v4_cache": {"path": str(cache_source), "sha256": observed_cache_sha},
        "teacher_replay_cache": {"path": str(source_path), "sha256": observed_source_sha},
        "contract": {
            "version": contract.version,
            "sha256": contract.digest,
            "maximum_tokens": contract.maximum_tokens,
            "token_candidate_limit": contract.token_candidate_limit,
            "core_token_fraction": contract.core_token_fraction,
        },
        "strict_checks": {
            "teacher_replay": teacher,
            "full_completion_pairing": "exact_teacher_identity_and_complete_variants_verified",
            "mask_partition": "core_context_support_fill_exactly_partition_token_mask",
            "padding": "student_and_teacher_padding_bitwise_zero",
            "feature_gauge": "active_radio_direction_unit_norm_with_fp16_storage_tolerance",
            "record_tensor_counts": "verified",
        },
        "membership_counts": counts,
        "query_free_context_risk_proxies": risk,
        "candidate_cap_audit_boundary": cap,
    }
    write_frozen_json(output, report)
    report["output"] = file_record(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--replay-cache", required=True)
    parser.add_argument("--replay-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_v4_stage1(
        args.cache,
        cache_sha256=args.cache_sha256,
        replay_cache_path=args.replay_cache,
        replay_cache_sha256=args.replay_cache_sha256,
        output=args.output,
    )
    print(report["output"])


if __name__ == "__main__":
    main()
