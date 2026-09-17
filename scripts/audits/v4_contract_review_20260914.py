"""Reproduce current v4 contract failures using production operators (CPU).
Run from repository root: PYTHONPATH=. python scripts/audits/v4_contract_review_20260914.py
These are diagnostic observations, not expected-correct regression tests.
"""
import json
import math
import torch
from radio_gs.v4.carrier.base import Camera
from radio_gs.v4.carrier.sparse_voxel import SurfaceVoxelCarrier
from radio_gs.v4.geometry.tsdf_fusion import DepthObservation, SparseSurfaceFusion
from radio_gs.v4.object_memory.sparse_assignment import SparseObjectAssignments
from radio_gs.v4.object_memory.fragment_to_object import (
    _pairwise_visibility_absence as _pairwise_visibility_conflict, _compose_observed_membership,
)


def carrier(points):
    return SurfaceVoxelCarrier(points, .04, maximum_splat_radius=0,
        surface_band_voxels=1.5, maximum_contributors_per_pixel=32,
        projection_contract="legacy_round_band_v1")


def main():
    camera = Camera('roundtrip', torch.eye(3), torch.eye(4), 1, 8)
    observation = DepthObservation(camera, torch.ones(1, 8),
        torch.ones(1, 8, dtype=torch.bool), torch.ones(1, 8))
    points, _, _, _ = SparseSurfaceFusion._backproject(observation)
    projection = carrier(points).project(camera)
    mapped = dict(zip(projection.element_ids.tolist(), projection.pixel_ids.tolist()))
    actual = [mapped.get(i) for i in range(8)]
    duplicate = {}
    for k in (1, 2, 4, 8, 16):
        assignments = SparseObjectAssignments.from_dense(torch.ones(1, k))
        duplicate[k] = float(assignments.mixture_sum(torch.ones(k))[0])
    noisy = {}
    for n in (1, 5, 20):
        noisy[n] = float(_compose_observed_membership(torch.full((n, 1), .05),
            torch.ones(n), torch.ones(n, 1), torch.zeros(n, dtype=torch.long), pooling="probabilistic_union")[0, 0])
    conflict = _pairwise_visibility_conflict(torch.eye(2), torch.arange(2),
        torch.ones(2, 2, dtype=torch.bool), evidence_threshold=.5)
    occlusion_camera = Camera('occlusion', torch.eye(3), torch.eye(4), 1, 1)
    occlusion = carrier(torch.tensor([[0., 0., 1.], [0., 0., 1.01]]))
    leaked = float(occlusion.render_posterior(torch.tensor([0., 1.]), occlusion_camera)[0, 0])
    fused = SparseSurfaceFusion(.04, minimum_views=2).fuse([observation, observation])
    result = {
        'pixel_roundtrip_actual': actual,
        'pixel_roundtrip_mismatches': sum(i != p for i, p in enumerate(actual)),
        'all_selected_duplicate_candidate_response': duplicate,
        'same_view_duplicate_evidence_response': noisy,
        'disjoint_visible_parts_conflict': float(conflict[0, 1]),
        'occluded_target_pixel_probability': leaked,
        'fused_unit_input_confidence_unique': fused.confidence.unique().tolist(),
        'maximum_possible_voxel_rms_over_size': math.sqrt(3) / 2,
        'default_dispersion_threshold_over_size': SparseSurfaceFusion(.04).maximum_dispersion_voxels,
        'sigmoid_margin_at_point_two': .03 * math.log(.2 / .8),
    }
    assert result['pixel_roundtrip_mismatches'] == 4
    assert duplicate[16] == .125
    assert noisy[20] > .64
    assert float(conflict[0, 1]) == 1
    assert leaked > .43
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
