"""Verify sealed D512+R5 deployment state reproduces source-dev capability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.training.instance_upper_bound import sha256_file, validate_source_only_inputs
from radio_gs.v3.training.run_instance_upper_bound import evaluate, load_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--expected-report", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state_path = Path(args.scene_state).resolve(strict=True)
    membership_path = Path(args.membership).resolve(strict=True)
    relation_path = Path(args.relation).resolve(strict=True)
    expected_path = Path(args.expected_report).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    relation = torch.load(relation_path, map_location="cpu")
    expected = json.loads(expected_path.read_text())
    validate_source_only_inputs(membership, relation)
    state_payload = torch.load(state_path, map_location="cpu")
    candidate_info = state_payload["metadata"]["inputs"]["candidate"]
    candidate_path = Path(candidate_info["path"]).resolve(strict=True)
    if (
        sha256_file(candidate_path) != candidate_info["sha256"]
        or candidate_info["sha256"] != expected["checkpoint"]["sha256"]
    ):
        raise ValueError("deployment state and selected checkpoint hash differ")
    candidate = torch.load(candidate_path, map_location="cpu")
    if not torch.equal(
        torch.as_tensor(state_payload["latent"]),
        torch.as_tensor(candidate["state_dict"]["memory"]),
    ):
        raise ValueError("deployment D512 differs from the selected checkpoint")
    for key, value in state_payload["global_state_dict"].items():
        if key not in candidate["state_dict"] or not torch.equal(
            torch.as_tensor(value), torch.as_tensor(candidate["state_dict"][key])
        ):
            raise ValueError("deployment global state differs from the selected checkpoint")
    interface = load_query_interface(state_path, device=args.device)
    episodes, supports = load_episodes(membership, relation)
    valid = [item for item in episodes if supports[item.proposal_index][0].numel()]
    training = [item for item in valid if item.view_index % 4 in (1, 2)]
    evaluation = [item for item in valid if item.view_index % 4 == 3]
    metrics, count = evaluate(
        interface.model,
        evaluation,
        supports,
        relation,
        {item.proposal_index for item in training},
        args.temperature,
        boundary_head=interface.boundary_head,
    )
    differences = {
        name: metrics[name] - expected["candidate_metrics"][name] for name in metrics
    }
    tolerances = {
        "mask_iou": 1e-8,
        "boundary_f": 1e-8,
        "brier": 1e-5,
        "unknown_fp_mass": 1e-5,
    }
    passed = count == expected["evaluation_proposals"] and all(
        abs(value) <= tolerances[name] for name, value in differences.items()
    )
    payload = {
        "schema": "radio_gs.sugm_v3.deployment_source_sentinel.v1",
        "scene": membership["scene"],
        "evaluation_proposals": count,
        "metrics": metrics,
        "delta_from_selected_checkpoint_report": differences,
        "metric_tolerances": tolerances,
        "deployment_tensors_exactly_equal_checkpoint": True,
        "same_gaussian_posterior_for_2d_and_3d": True,
        "pass": passed,
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            "scene_state": {"path": str(state_path), "sha256": sha256_file(state_path)},
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "relation": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
            "expected_report": {"path": str(expected_path), "sha256": sha256_file(expected_path)},
        },
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)
    if not passed:
        raise RuntimeError("deployment state does not reproduce the selected source-dev metrics")


if __name__ == "__main__":
    main()
