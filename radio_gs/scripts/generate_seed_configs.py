#!/usr/bin/env python3
"""Generate seeded config copies for confidence-interval runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from radio_gs.config import load_config, save_config


def parse_seeds(seed_text: str) -> list[int]:
    return [int(part.strip()) for part in seed_text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seeded RADIO-GS configs")
    parser.add_argument("configs", nargs="+", help="Base YAML configs to duplicate")
    parser.add_argument(
        "--seeds",
        default="7,42,123",
        help="Comma-separated seed list (default: 7,42,123)",
    )
    parser.add_argument(
        "--output_dir",
        default="radio_gs/configs/generated/seeds",
        help="Directory for generated YAML files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    for config_path in args.configs:
        base_path = Path(config_path)
        cfg = load_config(str(base_path))
        for seed in seeds:
            seeded = load_config(str(base_path))
            suffix = f"seed{seed}"
            seeded.seed = seed
            seeded.exp_name = f"{cfg.exp_name}_{suffix}"
            seeded.output_dir = f"{cfg.output_dir}_{suffix}"
            out_path = output_dir / f"{base_path.stem}_{suffix}.yaml"
            save_config(seeded, str(out_path))
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
