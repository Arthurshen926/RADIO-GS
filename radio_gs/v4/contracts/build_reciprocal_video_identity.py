"""Seal source-only video identity edges that agree in both directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.v4.contracts.geometry_receipt import sha256_file


def _unique_edges(pair: dict, minimum_iou: float) -> dict[int, dict]:
    best_by_target: dict[int, tuple[tuple[float, int], dict]] = {}
    for edge in pair["edges"]:
        iou = float(edge["tracked_to_target_root_iou"])
        target = int(edge["target_proposal_index"])
        if iou < minimum_iou or target < 0:
            continue
        key = (iou, -int(edge["source_proposal_index"]))
        if target not in best_by_target or key > best_by_target[target][0]:
            best_by_target[target] = (key, edge)
    return {
        int(value[1]["source_proposal_index"]): value[1]
        for value in best_by_target.values()
    }


def _reciprocal_edges(forward: dict, reverse: dict, minimum_iou: float) -> list[dict]:
    forward_edges = _unique_edges(forward, minimum_iou)
    reverse_edges = _unique_edges(reverse, minimum_iou)
    output = []
    for source_index, edge in sorted(forward_edges.items()):
        target_index = int(edge["target_proposal_index"])
        reverse_edge = reverse_edges.get(target_index)
        if reverse_edge is None or int(reverse_edge["target_proposal_index"]) != source_index:
            continue
        forward_iou = float(edge["tracked_to_target_root_iou"])
        reverse_iou = float(reverse_edge["tracked_to_target_root_iou"])
        output.append({
            "source_proposal_index": source_index,
            "target_proposal_index": target_index,
            "tracked_to_target_root_iou": min(forward_iou, reverse_iou),
            "forward_tracker_iou": forward_iou,
            "reverse_tracker_iou": reverse_iou,
            "accepted": True,
        })
    return output


def _load_pairs(paths: list[Path]) -> tuple[dict[tuple[int, int], dict], list[dict]]:
    pairs: dict[tuple[int, int], dict] = {}
    policies = []
    for path in paths:
        payload = json.loads(path.read_text())
        policy = payload.get("information_policy", {})
        if policy.get("benchmark_labels_used") is not False:
            raise ValueError("video identity input is not label-free")
        policies.append(policy)
        for pair in payload["pairs"]:
            key = (int(pair["source_frame_id"]), int(pair["target_frame_id"]))
            if key in pairs:
                raise ValueError(f"duplicate directed frame pair {key}")
            pairs[key] = pair
    return pairs, policies


def run(args: argparse.Namespace) -> dict:
    forward_paths = [Path(value).resolve(strict=True) for value in args.forward_manifest]
    reverse_paths = [Path(value).resolve(strict=True) for value in args.reverse_manifest]
    forward, _ = _load_pairs(forward_paths)
    reverse, _ = _load_pairs(reverse_paths)
    pair_key_path = (
        Path(args.pair_key_manifest).resolve(strict=True)
        if args.pair_key_manifest else None
    )
    if pair_key_path is not None:
        pair_key_records, _ = _load_pairs([pair_key_path])
        allowed_keys = set(pair_key_records)
        forward = {key: value for key, value in forward.items() if key in allowed_keys}
        if set(forward) != allowed_keys:
            raise ValueError("pair-key authority is not covered by forward manifests")
    records = []
    for key, pair in sorted(forward.items()):
        reverse_pair = reverse.get((key[1], key[0]))
        if reverse_pair is None:
            raise ValueError(f"missing reverse frame pair for {key}")
        edges = _reciprocal_edges(pair, reverse_pair, args.minimum_tracker_iou)
        records.append({
            "source_frame_id": key[0],
            "target_frame_id": key[1],
            "temporal_frame_gap": int(pair["temporal_frame_gap"]),
            "source_mask_cache": pair["source_mask_cache"],
            "target_mask_cache": pair["target_mask_cache"],
            "source_root_count": int(pair["source_root_count"]),
            "target_root_count": int(pair["target_root_count"]),
            "seeded_root_count": int(pair["seeded_root_count"]),
            "accepted_edge_count": len(edges),
            "edges": edges,
        })
    report = {
        "schema": "radio_gs.surface_object_memory_v4.reciprocal_video_identity.v1",
        "information_policy": {
            "source_rgb_used": True,
            "query_text_used": False,
            "benchmark_labels_used": False,
            "target_or_evaluation_rgb_used": False,
        },
        "identity_policy": {
            "agreement": "forward_and_reverse_proposal_indices_must_match",
            "minimum_tracker_iou_in_each_direction": args.minimum_tracker_iou,
            "duplicate_target_policy": "keep_highest_tracker_iou_then_lowest_source_index",
        },
        "inputs": [
            {"role": "forward_manifest", "path": str(path), "sha256": sha256_file(path)}
            for path in forward_paths
        ] + [
            {"role": "reverse_manifest", "path": str(path), "sha256": sha256_file(path)}
            for path in reverse_paths
        ] + ([{
            "role": "pair_key_manifest",
            "path": str(pair_key_path),
            "sha256": sha256_file(pair_key_path),
        }] if pair_key_path is not None else []),
        "pairs": records,
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-manifest", action="append", required=True)
    parser.add_argument("--reverse-manifest", action="append", required=True)
    parser.add_argument("--minimum-tracker-iou", type=float, default=0.70)
    parser.add_argument("--pair-key-manifest", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "pair_count": len(report["pairs"]),
        "reciprocal_edge_count": sum(row["accepted_edge_count"] for row in report["pairs"]),
    }, indent=2))


if __name__ == "__main__":
    main()
