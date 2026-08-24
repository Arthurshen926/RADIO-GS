#!/usr/bin/env python3
"""Run a same-latent A/B/C native-teacher pilot on one canonical field.

Arm A fits only a scene-global native decoder over the frozen L512 field.
Arm B jointly updates that same L512 table and decoder while preserving the
frozen RADIO decode.  Arm C performs the same update without the RADIO anchor.
No arm adds per-Gaussian state beyond the existing L512 table.  Selection uses
only disjoint held-out source views from the native-teacher cache; benchmark
queries, labels, masks, and metrics are not accepted by this entry point.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.field.checkpoint import load_factorized_canonical_field_checkpoint
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.native_multiteacher_abc_pilot.v2"


class NativeCapabilityDecoder(nn.Module):
    """One global decoder; it carries no primitive-local persistent state."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(latent_dim))
        self.hidden = nn.Linear(int(latent_dim), int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), int(output_dim))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        projected = self.output(F.gelu(self.hidden(self.norm(latent.float()))))
        return F.normalize(projected, dim=-1, eps=1e-8)


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _atomic_torch_save(value: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if output.exists():
            raise FileExistsError(f"refusing to overwrite immutable output: {output}")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists():
            raise FileExistsError(f"refusing to overwrite immutable output: {output}")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _coefficients_from_local(field: nn.Module, local: torch.Tensor) -> torch.Tensor:
    fusion = getattr(field, "fusion", None)
    if fusion is None:
        return local.to(dtype=field.decoder.basis.dtype)
    return fusion(local.to(dtype=next(fusion.parameters()).dtype), None, None)


def _radio_from_local(field: nn.Module, local: torch.Tensor) -> torch.Tensor:
    return field.decoder(_coefficients_from_local(field, local))


def _evaluate_native(
    decoder: nn.Module,
    local: torch.Tensor,
    rows: torch.Tensor,
    teacher: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    decoder.eval()
    cosines: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, rows.numel(), int(batch_size)):
            selected = rows[start : start + int(batch_size)]
            latent = local.index_select(0, selected.to(local.device))
            target = teacher.index_select(0, selected).to(device).float()
            predicted = decoder(latent)
            cosines.append(F.cosine_similarity(predicted, target, dim=-1).cpu())
    values = torch.cat(cosines)
    return {
        "mean_cosine": float(values.mean()),
        "p05_cosine": float(torch.quantile(values, 0.05)),
        "p50_cosine": float(torch.quantile(values, 0.50)),
        "rows": int(values.numel()),
    }


def _evaluate_radio(
    field: nn.Module,
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    rows: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    cosines: list[torch.Tensor] = []
    amplitude_errors: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, rows.numel(), int(batch_size)):
            selected = rows[start : start + int(batch_size)].to(candidate.device)
            current = _radio_from_local(field, candidate.index_select(0, selected))
            control = _radio_from_local(field, baseline.index_select(0, selected))
            cosines.append(F.cosine_similarity(current, control, dim=-1).cpu())
            amplitude_errors.append(
                (
                    current.norm(dim=-1).clamp_min(1e-8).log()
                    - control.norm(dim=-1).clamp_min(1e-8).log()
                )
                .abs()
                .cpu()
            )
    cosine = torch.cat(cosines)
    amplitude = torch.cat(amplitude_errors)
    return {
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(torch.quantile(cosine, 0.05)),
        "mean_abs_log_amplitude_error": float(amplitude.mean()),
        "p95_abs_log_amplitude_error": float(torch.quantile(amplitude, 0.95)),
        "rows": int(cosine.numel()),
    }


def _train_arm(
    *,
    name: str,
    field: nn.Module,
    baseline_local: torch.Tensor,
    initial_decoder_state: Mapping[str, torch.Tensor] | None,
    teacher_train: torch.Tensor,
    teacher_validation: torch.Tensor,
    train_rows: torch.Tensor,
    validation_rows: torch.Tensor,
    radio_probe_rows: torch.Tensor,
    device: torch.device,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    decoder_learning_rate: float,
    latent_learning_rate: float,
    radio_weight: float,
    train_latent: bool,
    seed: int,
    report_every: int,
) -> tuple[torch.Tensor, NativeCapabilityDecoder, dict[str, Any]]:
    _seed(seed)
    local = nn.Parameter(baseline_local.detach().clone(), requires_grad=train_latent)
    decoder = NativeCapabilityDecoder(
        int(local.shape[1]), int(hidden_dim), int(teacher_train.shape[1])
    ).to(device)
    if initial_decoder_state is not None:
        decoder.load_state_dict(initial_decoder_state, strict=True)
    decoder_optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=float(decoder_learning_rate), weight_decay=1e-4
    )
    latent_optimizer = (
        torch.optim.SparseAdam([local], lr=float(latent_learning_rate))
        if train_latent
        else None
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    history: list[dict[str, float | int]] = []
    decoder.train()
    for step in range(1, int(steps) + 1):
        selection = torch.randint(
            train_rows.numel(),
            (int(batch_size),),
            generator=generator,
        )
        rows = train_rows.index_select(0, selection)
        device_rows = rows.to(device)
        latent = F.embedding(device_rows, local, sparse=train_latent)
        target = teacher_train.index_select(0, rows).to(device).float()
        predicted = decoder(latent)
        native_loss = (1.0 - (predicted * target).sum(dim=-1)).mean()
        radio_loss = native_loss.new_zeros(())
        if train_latent and float(radio_weight) > 0:
            current_radio = _radio_from_local(field, latent)
            with torch.no_grad():
                baseline_radio = _radio_from_local(
                    field, baseline_local.index_select(0, device_rows)
                )
            direction = 1.0 - F.cosine_similarity(
                current_radio, baseline_radio, dim=-1
            ).mean()
            amplitude = F.smooth_l1_loss(
                current_radio.norm(dim=-1).clamp_min(1e-8).log(),
                baseline_radio.norm(dim=-1).clamp_min(1e-8).log(),
            )
            radio_loss = direction + amplitude
        loss = native_loss + float(radio_weight) * radio_loss
        decoder_optimizer.zero_grad(set_to_none=True)
        if latent_optimizer is not None:
            latent_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        decoder_optimizer.step()
        if latent_optimizer is not None:
            latent_optimizer.step()
        if step == 1 or step % int(report_every) == 0 or step == int(steps):
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "native_loss": float(native_loss.detach()),
                "radio_loss": float(radio_loss.detach()),
            }
            history.append(record)
            print(json.dumps({"arm": name, **record}), flush=True)
    native = _evaluate_native(
        decoder,
        local,
        validation_rows,
        teacher_validation,
        device=device,
        batch_size=int(batch_size) * 4,
    )
    radio = _evaluate_radio(
        field,
        local,
        baseline_local,
        radio_probe_rows,
        batch_size=int(batch_size) * 4,
    )
    return local.detach().cpu(), decoder, {
        "arm": name,
        "train_latent": bool(train_latent),
        "radio_weight": float(radio_weight),
        "history": history,
        "heldout_native": native,
        "radio_preservation": radio,
    }


def _candidate_field_payload(
    base_payload: Mapping[str, Any],
    *,
    local: torch.Tensor,
    decoder: NativeCapabilityDecoder,
    arm_report: Mapping[str, Any],
    base_field_path: Path,
    base_field_sha256: str,
    teacher_path: Path,
    teacher_sha256: str,
) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(base_payload))
    state = dict(candidate["state_dict"])
    if tuple(local.shape) != tuple(state["local_codes"].shape):
        raise ValueError("candidate native latent shape differs from base field")
    state["local_codes"] = local.float().contiguous()
    candidate["state_dict"] = state
    candidate["native_multiteacher"] = {
        "schema": SCHEMA,
        "base_field": {"path": str(base_field_path), "sha256": base_field_sha256},
        "teacher": {"path": str(teacher_path), "sha256": teacher_sha256},
        "persistent_primitive_state": "same_single_L512_local_codes_only",
        "global_native_decoder_state_dict": {
            key: value.detach().cpu() for key, value in decoder.state_dict().items()
        },
        "arm_report": dict(arm_report),
        "benchmark_queries_opened": False,
        "benchmark_ground_truth_opened": False,
    }
    return candidate


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = output_dir / "abc_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"A/B/C pilot already exists: {summary_path}")
    base_field_path = Path(args.base_field).expanduser().resolve(strict=True)
    teacher_path = Path(args.native_teacher).expanduser().resolve(strict=True)
    base_field_sha256 = sha256_file(base_field_path)
    teacher_sha256 = sha256_file(teacher_path)
    device = torch.device(args.device)
    field, base_payload, _signature = load_factorized_canonical_field_checkpoint(
        base_field_path, map_location=device, expected_sha256=base_field_sha256
    )
    field.eval().requires_grad_(False)
    teacher_payload = torch.load(teacher_path, map_location="cpu")
    if (
        not isinstance(teacher_payload, Mapping)
        or teacher_payload.get("schema")
        != "radio_gs.native_dinov2_exact_mpr_teacher.v1"
    ):
        raise ValueError("native teacher schema differs")
    teacher_train = torch.as_tensor(teacher_payload.get("features_train")).float()
    teacher_validation = torch.as_tensor(
        teacher_payload.get("features_validation")
    ).float()
    valid_train = torch.as_tensor(teacher_payload.get("valid_train")).bool()
    valid_validation = torch.as_tensor(
        teacher_payload.get("valid_validation")
    ).bool()
    if (
        teacher_train.ndim != 2
        or teacher_validation.shape != teacher_train.shape
        or valid_train.shape != (teacher_train.shape[0],)
        or valid_validation.shape != valid_train.shape
        or teacher_train.shape[0] != field.num_gaussians
        or not bool((valid_train & valid_validation).any())
    ):
        raise ValueError("native teacher tensors do not align with the field")
    train_rows = torch.where(valid_train)[0]
    validation_rows = torch.where(valid_train & valid_validation)[0]
    if int(args.maximum_validation_rows) > 0 and validation_rows.numel() > int(
        args.maximum_validation_rows
    ):
        generator = torch.Generator().manual_seed(int(args.seed) + 991)
        selected = torch.randperm(validation_rows.numel(), generator=generator)[
            : int(args.maximum_validation_rows)
        ]
        validation_rows = validation_rows.index_select(0, selected).sort().values
    radio_probe_rows = validation_rows
    baseline_local = field.local_codes.detach().clone()

    # Every arm must start from the exact same decoder parameters and consume
    # the exact same minibatch schedule.  Initializing B/C from the *trained* A
    # decoder would silently give them twice the decoder optimization budget
    # and confound the effect of updating the shared latent.
    _seed(int(args.seed))
    common_decoder = NativeCapabilityDecoder(
        int(baseline_local.shape[1]),
        int(args.hidden_dim),
        int(teacher_train.shape[1]),
    )
    common_initial_decoder_state = {
        key: value.detach().clone()
        for key, value in common_decoder.state_dict().items()
    }

    arm_a_local, arm_a_decoder, arm_a = _train_arm(
        name="A_frozen_radio_latent_global_native_decoder",
        field=field,
        baseline_local=baseline_local,
        initial_decoder_state=common_initial_decoder_state,
        teacher_train=teacher_train,
        teacher_validation=teacher_validation,
        train_rows=train_rows,
        validation_rows=validation_rows,
        radio_probe_rows=radio_probe_rows,
        device=device,
        hidden_dim=int(args.hidden_dim),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        decoder_learning_rate=float(args.decoder_learning_rate),
        latent_learning_rate=float(args.latent_learning_rate),
        radio_weight=0.0,
        train_latent=False,
        seed=int(args.seed),
        report_every=int(args.report_every),
    )
    arm_b_local, arm_b_decoder, arm_b = _train_arm(
        name="B_radio_anchored_native_multiteacher",
        field=field,
        baseline_local=baseline_local,
        initial_decoder_state=common_initial_decoder_state,
        teacher_train=teacher_train,
        teacher_validation=teacher_validation,
        train_rows=train_rows,
        validation_rows=validation_rows,
        radio_probe_rows=radio_probe_rows,
        device=device,
        hidden_dim=int(args.hidden_dim),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        decoder_learning_rate=float(args.decoder_learning_rate),
        latent_learning_rate=float(args.latent_learning_rate),
        radio_weight=float(args.radio_weight),
        train_latent=True,
        seed=int(args.seed),
        report_every=int(args.report_every),
    )
    arm_c_local, arm_c_decoder, arm_c = _train_arm(
        name="C_native_only_same_L512",
        field=field,
        baseline_local=baseline_local,
        initial_decoder_state=common_initial_decoder_state,
        teacher_train=teacher_train,
        teacher_validation=teacher_validation,
        train_rows=train_rows,
        validation_rows=validation_rows,
        radio_probe_rows=radio_probe_rows,
        device=device,
        hidden_dim=int(args.hidden_dim),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        decoder_learning_rate=float(args.decoder_learning_rate),
        latent_learning_rate=float(args.latent_learning_rate),
        radio_weight=0.0,
        train_latent=True,
        seed=int(args.seed),
        report_every=int(args.report_every),
    )
    arm_paths = {
        "A": output_dir / "arm_a_global_decoder.pt",
        "B": output_dir / "arm_b_radio_anchored_field.pth",
        "C": output_dir / "arm_c_native_only_field.pth",
    }
    _atomic_torch_save(
        {
            "schema": SCHEMA,
            "arm": "A",
            "base_field": {"path": str(base_field_path), "sha256": base_field_sha256},
            "native_teacher": {"path": str(teacher_path), "sha256": teacher_sha256},
            "decoder_state_dict": {
                key: value.detach().cpu()
                for key, value in arm_a_decoder.state_dict().items()
            },
            "report": arm_a,
        },
        arm_paths["A"],
    )
    _atomic_torch_save(
        _candidate_field_payload(
            base_payload,
            local=arm_b_local,
            decoder=arm_b_decoder,
            arm_report=arm_b,
            base_field_path=base_field_path,
            base_field_sha256=base_field_sha256,
            teacher_path=teacher_path,
            teacher_sha256=teacher_sha256,
        ),
        arm_paths["B"],
    )
    _atomic_torch_save(
        _candidate_field_payload(
            base_payload,
            local=arm_c_local,
            decoder=arm_c_decoder,
            arm_report=arm_c,
            base_field_path=base_field_path,
            base_field_sha256=base_field_sha256,
            teacher_path=teacher_path,
            teacher_sha256=teacher_sha256,
        ),
        arm_paths["C"],
    )
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "source_heldout_abc_complete_benchmark_not_opened",
        "base_field": {"path": str(base_field_path), "sha256": base_field_sha256},
        "native_teacher": {"path": str(teacher_path), "sha256": teacher_sha256},
        "method": {
            "same_geometry": True,
            "same_latent_size": int(baseline_local.shape[1]),
            "same_source_views": True,
            "per_gaussian_state": "one_L512_table",
            "native_decoder_scope": "scene_global_no_primitive_local_parameters",
            "radio_anchor_weight_B": float(args.radio_weight),
            "common_random_decoder_initialization": True,
            "matched_minibatch_schedule_across_arms": True,
            "matched_decoder_optimization_steps_across_arms": True,
        },
        "cohorts": {
            "train_rows": int(train_rows.numel()),
            "heldout_overlap_rows": int(validation_rows.numel()),
        },
        "arms": {"A": arm_a, "B": arm_b, "C": arm_c},
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in arm_paths.items()
        },
        "access_audit": {
            "source_rgb_teacher_opened": True,
            "benchmark_queries_opened": False,
            "benchmark_ground_truth_opened": False,
            "benchmark_metrics_opened": False,
        },
    }
    _atomic_json(summary, summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-field", required=True)
    parser.add_argument("--native-teacher", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--decoder-learning-rate", type=float, default=2e-4)
    parser.add_argument("--latent-learning-rate", type=float, default=2e-3)
    parser.add_argument("--radio-weight", type=float, default=1.0)
    parser.add_argument("--maximum-validation-rows", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=100)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
