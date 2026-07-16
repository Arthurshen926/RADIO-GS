#!/usr/bin/env python3
"""Aggregate frozen per-scene ScanNet text reports by class split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    protocols = [payload["protocol"] for payload in payloads]
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("ScanNet text protocols differ")
    splits = {}
    for split in payloads[0]["splits"]:
        rows = [payload["splits"][split] for payload in payloads]
        splits[split] = {
            "scene_macro_miou": sum(row["miou"] for row in rows) / len(rows),
            "scene_macro_macc": sum(row["macc"] for row in rows) / len(rows),
            "per_scene": {
                payload["scene"]: {"miou": row["miou"], "macc": row["macc"]}
                for payload, row in zip(payloads, rows)
            },
        }
    report = {"schema_version": 1, "protocol": protocols[0], "scene_count": len(payloads), "splits": splits}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
