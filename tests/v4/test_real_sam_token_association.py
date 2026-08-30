import json

import torch

from radio_gs.v4.carrier.base import Camera, ProjectionTable
from radio_gs.v4.evaluation.real_sam_token_association import (
    _lift_masks,
    _sealed_identity_labels,
)


class EmptyCarrier:
    num_elements = 2

    def project(self, camera):
        return ProjectionTable(
            element_ids=torch.tensor([0, 1]),
            pixel_ids=torch.tensor([0, 3]),
            depths=torch.ones(2),
            weights=torch.ones(2),
            num_elements=2,
            height=2,
            width=2,
        )


def test_empty_proposal_view_preserves_empty_evidence_shape():
    camera = Camera("empty", torch.eye(3), torch.eye(4), 2, 2)
    positive, visible = _lift_masks(EmptyCarrier(), camera, torch.empty(0, 2, 2), torch.device("cpu"))
    assert positive.shape == (0, 2)
    assert visible.shape == (0, 2)


def test_sealed_identity_edges_apply_threshold_and_resolve_target_collision(tmp_path):
    manifest = tmp_path / "identity_edges.json"
    manifest.write_text(json.dumps({
        "information_policy": {"benchmark_labels_used": False},
        "pairs": [{
            "source_frame_id": 10,
            "target_frame_id": 11,
            "edges": [
                {
                    "source_proposal_index": 1,
                    "target_proposal_index": 4,
                    "tracked_to_target_root_iou": 0.70,
                },
                {
                    "source_proposal_index": 2,
                    "target_proposal_index": 4,
                    "tracked_to_target_root_iou": 0.80,
                },
                {
                    "source_proposal_index": 3,
                    "target_proposal_index": 5,
                    "tracked_to_target_root_iou": 0.49,
                },
            ],
        }],
    }))

    labels = _sealed_identity_labels([manifest], minimum_tracker_iou=0.50)

    assert labels[10][2] == labels[11][4]
    assert 1 not in labels.get(10, {})
    assert 3 not in labels.get(10, {})
    assert 5 not in labels.get(11, {})
