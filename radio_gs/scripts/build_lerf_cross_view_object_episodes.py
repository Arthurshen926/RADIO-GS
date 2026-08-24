#!/usr/bin/env python3
"""Compile immutable LERF query-view -> target-view object episodes.

The compiler turns sparse DINO-cycle proposal relations into an explicit
training authority.  Same-object edges define connected object tracks;
different-object edges may provide negatives only in the target view.  Every
other Gaussian is unknown rather than an implicit negative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import _load_mapping, _proposal_support
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber


def compile_episodes(membership: dict[str, Any], authority: dict[str, Any], threshold: float) -> dict[str, Any]:
    rows = torch.as_tensor(membership["row_indices"]).long()
    proposals = torch.as_tensor(membership["proposal_indices"]).long()
    weights = torch.as_tensor(membership["weights"]).float()
    views = torch.as_tensor(membership["proposal_view_indices"]).long()
    count = int(membership["num_proposals"])
    authority_views = torch.as_tensor(authority["proposal_views"]).long()
    if views.shape != (count,) or not torch.equal(views, authority_views):
        raise ValueError("episode compiler proposal/view domain differs")
    hard = _proposal_support(rows[weights >= threshold], proposals[weights >= threshold], count)
    left = torch.as_tensor(authority["edge_left"]).long()
    right = torch.as_tensor(authority["edge_right"]).long()
    label = torch.as_tensor(authority["edge_label"]).long()
    if left.shape != right.shape or left.shape != label.shape:
        raise ValueError("episode compiler edge domain differs")

    parent = list(range(count))
    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    for a, b in zip(left[label == 1].tolist(), right[label == 1].tolist()):
        union(int(a), int(b))
    roots = sorted({find(i) for i in range(count)})
    root_to_object = {root: index for index, root in enumerate(roots)}

    relation: dict[tuple[int, int], int] = {}
    for a, b, value in zip(left.tolist(), right.tolist(), label.tolist()):
        relation[(int(a), int(b))] = relation[(int(b), int(a))] = int(value)
    query: list[int] = []
    target: list[int] = []
    object_id: list[int] = []
    negative_offsets = [0]
    negative_proposals: list[int] = []
    for a, b in zip(left[label == 1].tolist(), right[label == 1].tolist()):
        for q, t in ((int(a), int(b)), (int(b), int(a))):
            target_view = int(views[t])
            negatives = [
                candidate for candidate in range(count)
                if int(views[candidate]) == target_view
                and relation.get((q, candidate), -1) == 0
                and hard[candidate].numel() > 0
            ]
            if not negatives or hard[t].numel() == 0:
                continue
            query.append(q); target.append(t)
            object_id.append(root_to_object[find(q)])
            negative_proposals.extend(negatives)
            negative_offsets.append(len(negative_proposals))
    if not query:
        raise ValueError("episode compiler produced no supervised episodes")
    return {
        "schema": "radio_gs.lerf_cross_view_object_episodes.v1",
        "schema_version": 1,
        "episode_query_proposal": torch.tensor(query, dtype=torch.long),
        "episode_target_proposal": torch.tensor(target, dtype=torch.long),
        "episode_target_view": views[torch.tensor(target)],
        "episode_object_id": torch.tensor(object_id, dtype=torch.long),
        "negative_proposal_offsets": torch.tensor(negative_offsets, dtype=torch.long),
        "negative_proposals": torch.tensor(negative_proposals, dtype=torch.long),
        "metadata": {
            "source_only": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "positive_semantics": "DINO_cycle_same_object_target_proposal_membership",
            "negative_semantics": "explicit_DINO_cycle_different_instance_in_target_view_only",
            "unlisted_semantics": "unknown_excluded_from_loss",
            "membership_threshold": float(threshold),
            "episode_count": len(query),
            "object_track_count": len(set(object_id)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--expected-membership-sha256", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--membership-threshold", type=float, default=0.5)
    args = parser.parse_args()
    membership, membership_record = _load_mapping(args.membership, args.expected_membership_sha256, "source SAM membership")
    authority, authority_record = _load_mapping(args.authority, args.expected_authority_sha256, "source DINO physical authority")
    metadata = authority.get("metadata", {})
    if metadata.get("source_only") is not True or metadata.get("benchmark_masks_opened") is not False:
        raise ValueError("episode compiler information contract differs")
    payload = compile_episodes(membership, authority, args.membership_threshold)
    payload["metadata"]["inputs"] = {"membership": membership_record, "authority": authority_record}
    output = Path(args.output).expanduser().resolve()
    write_torch_noclobber(output, payload)
    report = {
        "status": "complete",
        "episodes": payload["metadata"]["episode_count"],
        "object_tracks": payload["metadata"]["object_track_count"],
        "output": file_record(output),
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
