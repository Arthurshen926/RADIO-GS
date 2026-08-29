"""Select per-scene masked semantic coverage using source dev only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _pair(value: str) -> tuple[Path, Path]:
    parts = value.split("::")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("scene must be DIRECT_REPORT::MASKED_REPORT")
    return tuple(Path(item).resolve(strict=True) for item in parts)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", type=_pair, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    selections = {}
    for direct_path, masked_path in args.scene:
        direct = json.loads(direct_path.read_text())
        masked = json.loads(masked_path.read_text())
        if direct["scene"] != masked["scene"] or direct["residue"] != 3 or masked["residue"] != 3:
            raise ValueError("masked semantic selection requires matched source-dev residue 3")
        before = direct["written_gaussian_d128"]
        after = masked["written_gaussian_d128"]
        accepted = (
            after["recall_at_1"] >= before["recall_at_1"]
            and after["mrr"] >= before["mrr"]
            and after["margin"] >= before["margin"]
        )
        selections[direct["scene"]] = {
            "selected": "masked_writer" if accepted else "direct_writer",
            "accepted": accepted,
            "direct": before, "masked": after,
            "inputs": {
                "direct": {"path": str(direct_path), "sha256": sha256_file(direct_path)},
                "masked": {"path": str(masked_path), "sha256": sha256_file(masked_path)},
            },
        }
    payload = {
        "schema": "radio_gs.sugm_v3.masked_semantic_writer_selection.v2",
        "selection_split": "source_dev_residue_3",
        "selection_rule": "nonregressing_recall_at_1_mrr_and_explicit_margin",
        "scenes": selections,
        "source_only": True, "audit_residue_opened_by_selector": False,
        "audit_residue_previously_opened_in_session": True,
        "selection_independence": "descriptive_dev_replay_not_pristine_audit",
        "benchmark_metrics_opened": False,
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()
