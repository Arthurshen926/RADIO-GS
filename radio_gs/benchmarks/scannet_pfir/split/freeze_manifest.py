"""Freeze an already constructed internal PFIR manifest."""

from __future__ import annotations

import argparse
import json

from radio_gs.benchmarks.scannet_pfir.protocol import ProtocolConfig, freeze_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-role", choices=("dev", "test"), required=True)
    args = parser.parse_args()
    payload = json.loads(open(args.input, encoding="utf-8").read())
    release = freeze_manifest(
        payload["queries"],
        args.output_dir,
        split_role=args.split_role,
        scene_reports=payload.get("scene_reports", []),
        config=ProtocolConfig(**payload.get("protocol_config", {})),
        selection_metadata=payload.get("scene_selection", {}),
    )
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
