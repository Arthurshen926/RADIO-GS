"""Immutable source-evidence authority for independently executed OOF folds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


SOURCE_EVIDENCE_TENSOR_NAMES = (
    "valid",
    "global_rows",
    "positive_weight",
    "negative_weight",
    "raw_positive_mass",
    "raw_negative_mass",
)
SOURCE_EVIDENCE_REPLAY_RTOL = 1e-4


@dataclass(frozen=True)
class SourceObservationEvidenceAuthority:
    """Bitwise-stable prompt evidence shared by every OOF compiler process."""

    path: Path
    sha256: str
    content_sha256: str
    tensors: dict[str, torch.Tensor]
    replay_max_relative_error: dict[str, float]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _canonical_source_evidence_tensors(
    *,
    valid: torch.Tensor,
    global_rows: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
    raw_positive_mass: torch.Tensor,
    raw_negative_mass: torch.Tensor,
) -> dict[str, torch.Tensor]:
    tensors = {
        "valid": torch.as_tensor(valid).detach().bool().cpu().reshape(-1),
        "global_rows": torch.as_tensor(global_rows)
        .detach()
        .long()
        .cpu()
        .reshape(-1),
        "positive_weight": torch.as_tensor(positive_weight)
        .detach()
        .float()
        .cpu()
        .reshape(-1),
        "negative_weight": torch.as_tensor(negative_weight)
        .detach()
        .float()
        .cpu()
        .reshape(-1),
        "raw_positive_mass": torch.as_tensor(raw_positive_mass)
        .detach()
        .float()
        .cpu()
        .reshape(-1),
        "raw_negative_mass": torch.as_tensor(raw_negative_mass)
        .detach()
        .float()
        .cpu()
        .reshape(-1),
    }
    valid_cpu = tensors["valid"]
    rows = tensors["global_rows"]
    if valid_cpu.numel() == 0 or not torch.equal(rows, torch.where(valid_cpu)[0]):
        raise ValueError(
            "source-evidence authority rows differ from sorted valid-row authority"
        )
    for name in SOURCE_EVIDENCE_TENSOR_NAMES[2:]:
        value = tensors[name]
        if value.shape != valid_cpu.shape:
            raise ValueError(
                f"source-evidence authority tensor {name!r} does not align"
            )
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(
                f"source-evidence authority tensor {name!r} is invalid"
            )
    return tensors


def _validate_sealed_payload(
    payload: object,
    *,
    provenance: Mapping[str, object],
) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(payload, Mapping):
        raise ValueError("source-evidence authority is not a mapping")
    if payload.get("artifact_type") != "source_observation_evidence_authority_v1":
        raise ValueError("source-evidence authority artifact type differs")
    if payload.get("provenance") != dict(provenance):
        raise ValueError("source-evidence authority provenance differs")
    tensors_payload = payload.get("tensors")
    hashes = payload.get("tensor_sha256")
    if not isinstance(tensors_payload, Mapping) or set(tensors_payload) != set(
        SOURCE_EVIDENCE_TENSOR_NAMES
    ):
        raise ValueError("source-evidence authority tensor schema differs")
    if not isinstance(hashes, Mapping) or set(hashes) != set(
        SOURCE_EVIDENCE_TENSOR_NAMES
    ):
        raise ValueError("source-evidence authority tensor hashes differ")
    tensors = _canonical_source_evidence_tensors(
        **{name: tensors_payload[name] for name in SOURCE_EVIDENCE_TENSOR_NAMES}
    )
    for name, value in tensors.items():
        if str(hashes[name]) != _tensor_sha256(value):
            raise ValueError(f"source-evidence authority tensor {name!r} changed")
    contract = {
        "schema_version": 1,
        "artifact_type": "source_observation_evidence_authority_v1",
        "provenance": dict(provenance),
        "tensor_sha256": {name: str(hashes[name]) for name in hashes},
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    content_sha256 = _json_sha256(contract)
    if str(payload.get("content_sha256")) != content_sha256:
        raise ValueError("source-evidence authority content hash differs")
    if any(
        bool(payload.get(name, True))
        for name in (
            "target_rgb_opened",
            "target_mask_opened",
            "target_metric_computed",
        )
    ):
        raise ValueError("source-evidence authority was not sealed before target access")
    return tensors, content_sha256


def seal_or_load_source_observation_evidence_authority(
    path: str | Path,
    *,
    heldout_fold: int,
    provenance: Mapping[str, object],
    valid: torch.Tensor,
    global_rows: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
    raw_positive_mass: torch.Tensor,
    raw_negative_mass: torch.Tensor,
) -> SourceObservationEvidenceAuthority:
    """Seal fold-0 evidence once, then replay it bitwise for every OOF fold.

    Exact raster-adjoint values can differ in their low CUDA floating-point bits
    across otherwise identical processes.  The local recomputation is therefore
    used only as an attestation: support must match exactly and nonzero values
    must agree to the fixed float32 replay tolerance.  Every compiler consumes
    the sealed tensors, never the jittered local copy.
    """

    fold = int(heldout_fold)
    if fold not in (0, 1, 2):
        raise ValueError("source-evidence authority fold must be 0, 1, or 2")
    if not provenance or any(not str(key) for key in provenance):
        raise ValueError("source-evidence authority provenance is empty")
    local = _canonical_source_evidence_tensors(
        valid=valid,
        global_rows=global_rows,
        positive_weight=positive_weight,
        negative_weight=negative_weight,
        raw_positive_mass=raw_positive_mass,
        raw_negative_mass=raw_negative_mass,
    )
    output = Path(path).expanduser().resolve()
    if not output.exists():
        if fold != 0:
            raise FileNotFoundError(
                "source-evidence authority must be sealed by fold 0 before later folds"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        tensor_hashes = {name: _tensor_sha256(value) for name, value in local.items()}
        contract = {
            "schema_version": 1,
            "artifact_type": "source_observation_evidence_authority_v1",
            "provenance": dict(provenance),
            "tensor_sha256": tensor_hashes,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_computed": False,
        }
        payload = {
            **contract,
            "content_sha256": _json_sha256(contract),
            "tensors": local,
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        if temporary.exists():
            raise FileExistsError(
                f"source-evidence authority temporary path already exists: {temporary}"
            )
        torch.save(payload, temporary)
        temporary.replace(output)

    payload = torch.load(output, map_location="cpu", weights_only=False)
    sealed, content_sha256 = _validate_sealed_payload(
        payload,
        provenance=provenance,
    )
    if not torch.equal(local["valid"], sealed["valid"]) or not torch.equal(
        local["global_rows"], sealed["global_rows"]
    ):
        raise ValueError("local source-evidence row authority differs from sealed authority")
    replay_error: dict[str, float] = {}
    for name in SOURCE_EVIDENCE_TENSOR_NAMES[2:]:
        candidate = local[name]
        reference = sealed[name]
        if not torch.equal(candidate == 0, reference == 0):
            raise ValueError(
                f"local source-evidence support {name!r} differs from sealed authority"
            )
        if not torch.allclose(
            candidate,
            reference,
            rtol=SOURCE_EVIDENCE_REPLAY_RTOL,
            atol=0.0,
        ):
            raise ValueError(
                f"local source-evidence tensor {name!r} exceeds replay tolerance"
            )
        nonzero = reference != 0
        replay_error[name] = (
            float(
                ((candidate[nonzero] - reference[nonzero]).abs() / reference[nonzero].abs())
                .max()
                .item()
            )
            if bool(nonzero.any())
            else 0.0
        )
    return SourceObservationEvidenceAuthority(
        path=output,
        sha256=_file_sha256(output),
        content_sha256=content_sha256,
        tensors=sealed,
        replay_max_relative_error=replay_error,
    )
