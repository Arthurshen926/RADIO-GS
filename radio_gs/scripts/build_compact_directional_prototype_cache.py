#!/usr/bin/env python3
"""Compress a query-free directional prototype bank in a frozen field basis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from radio_gs.field import (
    AffineBasisDecoder,
    DIRECTIONAL_PROTOTYPE_CONTRACT,
    load_canonical_field_checkpoint,
)
from radio_gs.utils.immutable_artifacts import load_torch_mapping, sha256_file


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@torch.no_grad()
def build(args: argparse.Namespace) -> dict[str, object]:
    prototype, prototype_sha, prototype_path = load_torch_mapping(
        args.prototype_cache,
        expected_sha256=args.expected_prototype_cache_sha256,
        map_location="cpu",
        label="Field-D directional prototype cache",
    )
    if int(prototype.get("schema_version", -1)) != 1 or prototype.get(
        "contract"
    ) != DIRECTIONAL_PROTOTYPE_CONTRACT:
        raise ValueError("Field-D prototype contract differs")
    metadata = dict(prototype.get("metadata", {}))
    if any(
        metadata.get(key) is not False
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
        )
    ):
        raise ValueError("Field-D prototype cache is task contaminated")
    modes = torch.as_tensor(prototype["prototypes"]).float().cpu()
    rows = torch.as_tensor(prototype["global_rows"]).long().cpu()
    mixture = torch.as_tensor(prototype["mixture_weight"]).float().cpu()
    if modes.ndim != 3 or modes.shape[1] != 2 or modes.shape[2] != 1280:
        raise ValueError("Field-D prototype tensor shape differs")
    if rows.shape != (modes.shape[0],) or mixture.shape != modes.shape[:2]:
        raise ValueError("Field-D prototype row metadata does not align")
    if rows.numel() and (
        not bool((rows[1:] > rows[:-1]).all()) or int(rows.min()) < 0
    ):
        raise ValueError("Field-D prototype global rows are not strictly ascending")

    basis_authority: dict[str, str]
    if str(args.joint_basis_checkpoint).strip():
        joint, joint_sha, joint_path = load_torch_mapping(
            args.joint_basis_checkpoint,
            expected_sha256=args.expected_joint_basis_checkpoint_sha256,
            map_location="cpu",
            label="Field-D joint directional basis",
        )
        joint_contract = joint.get("contract")
        if joint_contract not in {
            "field_d_joint_center_directional_basis_v1",
            "field_d_gauge_preserving_joint_basis_v1",
        }:
            raise ValueError("Field-D joint basis contract differs")
        if joint.get("geometry_fingerprint") != prototype.get(
            "geometry_fingerprint"
        ):
            raise ValueError("Field-D prototype/joint-basis geometry differs")
        architecture = dict(joint["architecture"])
        state = dict(joint["decoder_state_dict"])
        decoder = AffineBasisDecoder(
            feature_dim=int(architecture["feature_dim"]),
            coefficient_dim=int(architecture["coefficient_dim"]),
            mean=state["mean"],
            scale=torch.as_tensor(state["log_scale"]).exp(),
            basis=state["basis"],
            trainable_basis=False,
        )
        basis_authority = {"path": str(joint_path), "sha256": joint_sha}
        num_gaussians = int(
            prototype["geometry_fingerprint"]["num_gaussians"]
        )
        if joint_contract == "field_d_gauge_preserving_joint_basis_v1":
            amplitude_rows = torch.as_tensor(joint["prototype_global_rows"]).long()
            amplitude = torch.as_tensor(joint["prototype_amplitude"]).float()
            if not torch.equal(amplitude_rows, rows) or amplitude.shape != (
                rows.numel(),
            ):
                raise ValueError("gauge-preserving amplitude rows differ")
            if not bool(torch.isfinite(amplitude).all()) or bool(
                (amplitude <= 1e-8).any()
            ):
                raise ValueError("gauge-preserving amplitudes are invalid")
            modes = F.normalize(modes, dim=-1, eps=1e-8) * amplitude[:, None, None]
    else:
        if not str(args.field_checkpoint).strip() or not str(
            args.expected_field_checkpoint_sha256
        ):
            raise ValueError("a field or joint basis checkpoint is required")
        field_path = Path(args.field_checkpoint).expanduser().resolve()
        field, field_payload = load_canonical_field_checkpoint(
            field_path,
            map_location="cpu",
            expected_sha256=args.expected_field_checkpoint_sha256,
        )
        if prototype.get("geometry_fingerprint") != field_payload.get(
            "geometry_fingerprint"
        ):
            raise ValueError("Field-D prototype/field geometry differs")
        decoder = field.decoder
        num_gaussians = int(field.num_gaussians)
        basis_authority = {
            "path": str(field_path),
            "sha256": args.expected_field_checkpoint_sha256,
        }
    if rows.numel() and int(rows.max()) >= num_gaussians:
        raise ValueError("Field-D prototype global row exceeds carrier geometry")
    if decoder.feature_dim != 1280:
        raise ValueError("Field-D compact carrier requires RADIO-1280 decoder")

    device = torch.device(args.device)
    decoder = decoder.to(device).eval()
    coefficients = torch.empty(
        modes.shape[0], 2, decoder.coefficient_dim, dtype=torch.float16
    )
    cosine_parts: list[torch.Tensor] = []
    for start in range(0, modes.shape[0], int(args.batch_size)):
        stop = min(start + int(args.batch_size), modes.shape[0])
        target = modes[start:stop].to(device)
        encoded = decoder.encode(target)
        reconstructed = decoder(encoded)
        cosine_parts.append(
            F.cosine_similarity(reconstructed, target, dim=-1, eps=1e-8)
            .float()
            .cpu()
        )
        coefficients[start:stop] = encoded.half().cpu()
    cosine = torch.cat(cosine_parts).reshape(-1)
    metrics = {
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(torch.quantile(cosine, 0.05)),
        "minimum_cosine": float(cosine.min()),
    }
    gate = {
        "mean_cosine": metrics["mean_cosine"]
        >= float(args.minimum_mean_cosine),
        "p05_cosine": metrics["p05_cosine"] >= float(args.minimum_p05_cosine),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "contract": "compact_directional_prototype_field_v1",
            "global_rows": rows,
            "coefficients": coefficients,
            "mixture_weight": mixture.half(),
            "observation_count": torch.as_tensor(
                prototype["observation_count"]
            ).to(torch.int16),
            "center_resultant": torch.as_tensor(
                prototype["center_resultant"]
            ).half(),
            "geometry_fingerprint": prototype["geometry_fingerprint"],
            "metadata": {
                "source_prototype_cache": str(prototype_path),
                "source_prototype_cache_sha256": prototype_sha,
                "basis_authority": basis_authority,
                "coefficient_dim": decoder.coefficient_dim,
                "feature_dim": decoder.feature_dim,
                "compression_metrics": metrics,
                "feature_gauge": (
                    "per_primitive_base_field_l2_norm"
                    if str(args.joint_basis_checkpoint).strip()
                    and joint.get("contract")
                    == "field_d_gauge_preserving_joint_basis_v1"
                    else "source_prototype_native"
                ),
                "gate": gate,
                "benchmark_images_opened": False,
                "benchmark_masks_opened": False,
                "text_queries_opened": False,
            },
        },
        output,
    )
    output_sha = sha256_file(output)
    report: dict[str, object] = {
        "schema_version": "compact_directional_prototype_field_receipt_v1",
        "output": {"path": str(output), "sha256": output_sha},
        "source_prototype_cache": {
            "path": str(prototype_path),
            "sha256": prototype_sha,
        },
        "basis_authority": basis_authority,
        "rows": int(rows.numel()),
        "prototype_count": 2,
        "coefficient_dim": int(decoder.coefficient_dim),
        "metrics": metrics,
        "gate": gate,
        "passed": bool(all(gate.values())),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    receipt = output.with_suffix(output.suffix + ".json")
    _atomic_json(receipt, report)
    return {**report, "receipt": str(receipt), "receipt_sha256": sha256_file(receipt)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-cache", required=True)
    parser.add_argument("--expected-prototype-cache-sha256", required=True)
    parser.add_argument("--field-checkpoint", default="")
    parser.add_argument("--expected-field-checkpoint-sha256", default="")
    parser.add_argument("--joint-basis-checkpoint", default="")
    parser.add_argument("--expected-joint-basis-checkpoint-sha256", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--minimum-mean-cosine", type=float, default=0.95)
    parser.add_argument("--minimum-p05-cosine", type=float, default=0.90)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
