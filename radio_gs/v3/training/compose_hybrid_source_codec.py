"""Compose the fixed-JL visual control with the learned semantic codec."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.structured_initialization import fixed_jl_projection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-codec", required=True)
    parser.add_argument("--radio-seed", type=int, default=20260826)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parent_path = Path(args.parent_codec).resolve(strict=True)
    parent = torch.load(parent_path, map_location="cpu")
    state = parent["state_dict"]
    input_dim, output_dim = torch.as_tensor(state["radio_basis"]).shape
    payload = {
        "schema": parent["schema"],
        "state_dict": {
            **state,
            "radio_mean": torch.zeros(input_dim),
            "radio_basis": fixed_jl_projection(input_dim, output_dim, args.radio_seed),
        },
        "metadata": {
            **parent["metadata"],
            "type": "fixed_jl_visual_learned_semantic_exact_mpr_order_control",
            "parent_codec": {"path": str(parent_path), "sha256": sha256_file(parent_path)},
            "radio": {
                **parent["metadata"]["radio"],
                "projection_type": "fixed_jl",
                "seed": args.radio_seed,
                "retained_variance_fraction": None,
            },
            "normalization_order": "linear_projection_then_exact_mpr_then_row_normalize",
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    print({"output": str(output), "sha256": sha256_file(output)})


if __name__ == "__main__":
    main()
