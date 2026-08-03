import argparse
import copy

import pytest
import torch

from radio_gs.querying.adaptive_support import (
    AdaptiveSupportCalibrationError,
    exact_otsu_threshold,
    select_adaptive_otsu_support,
)
from radio_gs.scripts.eval_lerf_adaptive_support_diagnostic import (
    _load_cache_inputs,
    authority_tensor_sha256,
    build_frozen_evaluator_argv,
    precompute_adaptive_membership,
)
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen_evaluator
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


def _authority_bound_cache_payload() -> dict[str, object]:
    xyz = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    scores = torch.linspace(0.0, 1.0, 36, dtype=torch.float16).reshape(6, 3, 2)
    valid = torch.tensor([True, True, False, True, True, True])
    query_ids = ["red cup", "tea pot"]
    scale_radii = [0.25, 0.45, 0.7]
    scale_ids = [str(value) for value in scale_radii]
    field_sha = "b" * 64
    readout_sha = "c" * 64
    renderer_sha = "a" * 64
    xyz_fingerprint = frozen_evaluator.xyz_geometry_fingerprint(xyz)
    xyz_sha = xyz_fingerprint["xyz_sha256"]
    authority = {
        "contract": frozen_evaluator.OURS_MULTISCALE_QUERY_SCORE_AUTHORITY_CONTRACT,
        "score_semantics": "raw_independent_normalized_cosine",
        "score_dtype": "torch.float16",
        "scale_axis": [
            {"id": scale_id, "value": radius, "unit": "meter"}
            for scale_id, radius in zip(scale_ids, scale_radii)
        ],
        "query_axis": {
            "ids": query_ids,
            "order_sha256": canonical_json_sha256(query_ids),
        },
        "geometry_axis": {
            "num_gaussians": len(xyz),
            "xyz_sha256": xyz_sha,
            "renderer_xyz_sha256": xyz_sha,
            "valid_sha256": authority_tensor_sha256(valid),
            "field_checkpoint_sha256": field_sha,
            "readout_checkpoint_sha256": readout_sha,
            "renderer_geometry_checkpoint_sha256": renderer_sha,
        },
        "query_scores_sha256": authority_tensor_sha256(scores),
        "source_artifacts": {
            "field_checkpoint": {"path": "/frozen/field.pt", "sha256": field_sha},
            "readout_checkpoint": {
                "path": "/frozen/readout.pt",
                "sha256": readout_sha,
            },
            "renderer_geometry_checkpoint": {
                "path": "/frozen/renderer.pt",
                "sha256": renderer_sha,
            },
        },
        "calibration_constraints": {
            "softmax_applied": False,
            "temperature_applied": False,
            "peak_normalization_applied": False,
            "threshold_applied": False,
            "scale_reduction_applied": False,
            "benchmark_images_opened": False,
            "benchmark_annotations_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
        },
    }
    return {
        "version": frozen_evaluator.OURS_MULTISCALE_QUERY_SCORE_CACHE_VERSION,
        "contract": frozen_evaluator.OURS_MULTISCALE_QUERY_SCORE_CACHE_CONTRACT,
        "query_scores": scores,
        "query_ids": query_ids,
        "scale_ids": scale_ids,
        "scale_radii_m": scale_radii,
        "xyz": xyz,
        "valid": valid,
        "geometry_fingerprint": xyz_fingerprint,
        "field_checkpoint_sha256": field_sha,
        "readout_checkpoint_sha256": readout_sha,
        "renderer_geometry_checkpoint_sha256": renderer_sha,
        "authority": authority,
    }


def test_exact_otsu_splits_two_separated_classes_without_histogram_bins() -> None:
    threshold = exact_otsu_threshold(torch.tensor([0.0, 0.1, 0.2, 0.8, 0.9, 1.0]))
    assert threshold == pytest.approx(0.5)


def test_recursive_upper_otsu_is_stricter_and_invalid_rows_never_select() -> None:
    scores = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.3, 0.4],
            [0.6, 0.7],
            [0.8, 0.9],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    valid = torch.tensor([True, True, True, True, True, True, False])
    first = select_adaptive_otsu_support(scores, valid, otsu_stages=1)
    third = select_adaptive_otsu_support(scores, valid, otsu_stages=3)

    assert torch.all(third.thresholds >= first.thresholds)
    assert torch.all(third.selected <= first.selected)
    assert torch.equal(third.selected[-1], torch.zeros(2))
    assert torch.equal(third.selected_counts, torch.ones(2, dtype=torch.long))


def test_otsu_fails_closed_on_degenerate_scores() -> None:
    with pytest.raises(AdaptiveSupportCalibrationError, match="distinct"):
        select_adaptive_otsu_support(
            torch.ones(8, 2),
            torch.ones(8, dtype=torch.bool),
            otsu_stages=1,
        )


def test_wrapper_builds_only_frozen_vala_repo_invocation() -> None:
    args = argparse.Namespace(
        config="config.yaml",
        checkpoint="geometry.pth",
        scene="figurines",
        label_dir="labels",
        output_dir="diagnostic",
        summary_head_weights="summary.pth",
        text_embedding_cache="queries.pt",
        canonical_embedding_cache="negatives.pt",
        ours_multiscale_query_score_cache="scores.pt",
        gpu=0,
    )
    argv = build_frozen_evaluator_argv(args)

    assert argv[argv.index("--protocol_preset") + 1] == "vala_repo_3d"
    assert "--threshold_sweep" not in argv
    assert "--score_threshold" not in argv
    assert "--selection_refinement" not in argv
    assert "--mask_refinement" not in argv
    assert argv[argv.index("--ours_multiscale_query_score_cache") + 1] == "scores.pt"


def test_precomputed_membership_is_hashed_before_evaluator_use(monkeypatch) -> None:
    scores = torch.tensor(
        [
            [[0.0], [0.0], [0.0]],
            [[0.1], [0.2], [0.3]],
            [[0.4], [0.5], [0.6]],
            [[0.7], [0.8], [0.9]],
            [[1.0], [1.0], [1.0]],
        ]
    )
    expected_processed = torch.tensor([[0.0], [0.2], [0.4], [0.8], [1.0]])

    class Readout:
        def __init__(self, values):
            self.scores = values

    monkeypatch.setattr(
        "radio_gs.scripts.eval_lerf_adaptive_support_diagnostic."
        "frozen_evaluator.vala_multiscale_knn_peak_select_scores",
        lambda *args, **kwargs: Readout(expected_processed),
    )
    cache = {
        "query_scores": scores,
        "xyz": torch.zeros(5, 3),
        "valid": torch.ones(5, dtype=torch.bool),
    }
    result = precompute_adaptive_membership(cache, otsu_stages=1)

    assert result["membership_sha256"] == authority_tensor_sha256(
        result["selection"].selected.bool()
    )
    assert result["processed_scores_sha256"]


def test_precompute_loader_accepts_complete_frozen_authority(tmp_path) -> None:
    path = tmp_path / "scores.pt"
    torch.save(_authority_bound_cache_payload(), path)

    loaded = _load_cache_inputs(path)

    assert loaded["query_ids"] == ("red cup", "tea pot")
    assert loaded["query_scores"].dtype == torch.float32
    assert loaded["renderer_geometry_checkpoint_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "tamper,match",
    [
        ("calibration", "calibration constraints differ"),
        ("semantics", "score semantics differ"),
        ("declared_dtype", "declared score dtype differs"),
        ("tensor_dtype", "score tensor dtype differs"),
        ("query_order", "query-order hash mismatch"),
        ("xyz", "fingerprint does not match"),
    ],
)
def test_precompute_loader_rejects_incomplete_or_tampered_authority(
    tmp_path,
    tamper: str,
    match: str,
) -> None:
    payload = copy.deepcopy(_authority_bound_cache_payload())
    authority = payload["authority"]
    assert isinstance(authority, dict)
    if tamper == "calibration":
        authority["calibration_constraints"].pop("threshold_applied")
    elif tamper == "semantics":
        authority["score_semantics"] = "post_threshold_scores"
    elif tamper == "declared_dtype":
        authority["score_dtype"] = "torch.float32"
    elif tamper == "tensor_dtype":
        payload["query_scores"] = payload["query_scores"].float()
        authority["query_scores_sha256"] = authority_tensor_sha256(
            payload["query_scores"]
        )
    elif tamper == "query_order":
        authority["query_axis"]["order_sha256"] = "0" * 64
    elif tamper == "xyz":
        payload["xyz"][0, 0] += 1.0
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(tamper)
    path = tmp_path / f"{tamper}.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match=match):
        _load_cache_inputs(path)
