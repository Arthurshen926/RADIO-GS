import hashlib
import json
from pathlib import Path

import torch

from radio_gs.field import FeatureSpaceSignature
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.primitive_row_authority import PrimitiveRowAuthority
from radio_gs.scripts.build_exact_mpr_capability_views import (
    EXACT_MPR_CAPABILITY_SOURCE,
    _exact_signature,
    _feature_rows,
)
from radio_gs.scripts.build_exact_capability_mpr_views import _pair_sha256
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _disabled_registered_graph,
    _write_pre_metric_prediction_receipt,
    _write_primitive_unary_artifact,
)
from radio_gs.scripts.summarize_nvos_exact_compact_ladder import _difference


def _signature(dim: int, field_hash: str) -> dict:
    return FeatureSpaceSignature(
        radio_version="c-radio_v4-h",
        radio_checkpoint_sha256="a" * 64,
        raw_feature_dim=4,
        adaptor_name=f"head{dim}.feature_projection",
        adaptor_sha256="a" * 64,
        adaptor_output_dim=dim,
        token_type="primitive",
        normalization="l2",
        crop_policy="training_views_depth_alpha_checked_mpr",
        field_checkpoint_sha256=field_hash,
    ).to_dict()


def _write_exact_capability(path: Path) -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    valid = torch.tensor([True, False])
    digest = "b" * 64
    metadata = {
        "source": EXACT_MPR_CAPABILITY_SOURCE,
        "exact_raw_mpr_sha256": digest,
        "field_checkpoint_sha256": digest,
        "custom_adaptor_head": False,
        "query_independent": True,
        "feature_storage": "valid_rows_compact_v1",
        "feature_row_order": "torch_where_valid_ascending",
        "feature_row_count": 1,
        "primitive_row_authority": PrimitiveRowAuthority.from_tensors(
            xyz, valid
        ).to_dict(),
        "projection_contract": {
            "contract": "radio_gs.raw_mpr_then_nonlinear_adaptor_diagnostic.v1",
            "eligibility": "diagnostic_only",
            "projection_order": "raw_radio_mpr_then_official_adaptor",
            "query_dependent": False,
        },
        "capability_signatures": {
            "appearance": _signature(3, digest),
            "boundary": _signature(2, digest),
        },
    }
    torch.save(
        {
            "schema_version": 1,
            "xyz": xyz,
            "valid": valid,
            "appearance_dino_v3": torch.tensor([[1.0, 0.0, 0.0]]).half(),
            "boundary_sam3": torch.tensor([[0.0, 1.0]]).half(),
            "metadata": metadata,
        },
        path,
    )


def test_exact_capability_requires_explicit_source_contract(tmp_path: Path) -> None:
    cache = tmp_path / "exact.pt"
    _write_exact_capability(cache)
    try:
        load_canonical_capability_bank(cache)
    except ValueError as error:
        assert "source contract differs" in str(error)
    else:
        raise AssertionError("exact diagnostic cache passed as a canonical field")
    bank = load_canonical_capability_bank(
        cache,
        expected_source=EXACT_MPR_CAPABILITY_SOURCE,
        require_row_authority=True,
        allow_raw_mpr_projection_diagnostic=True,
    )
    assert bank.features_are_compact is True
    assert bank.appearance.shape == (1, 3)


def test_exact_signature_never_claims_compact_field_origin() -> None:
    reference = FeatureSpaceSignature.from_mapping(_signature(3, "c" * 64))
    exact = _exact_signature(reference, exact_mpr_sha256="d" * 64)
    assert exact.field_checkpoint_sha256 == "d" * 64
    exact.assert_comparable(reference)


def test_exact_capability_pair_digest_is_order_and_role_sensitive() -> None:
    forward = _pair_sha256("a" * 64, "b" * 64)
    reverse = _pair_sha256("b" * 64, "a" * 64)
    assert len(forward) == 64
    assert forward != reverse


def test_feature_rows_uses_dense_exact_mpr_rows() -> None:
    features = torch.arange(20).reshape(5, 4).half()
    selected = _feature_rows({"features": features}, torch.tensor([3, 1]))
    assert torch.equal(selected, features[torch.tensor([3, 1])])


def test_disabled_graph_has_no_edges_or_channels() -> None:
    graph = _disabled_registered_graph(7)
    assert graph.num_nodes == 7
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_weight.numel() == 0
    assert graph.edge_channels == {}


def test_primitive_unary_artifact_is_dense_pre_gt_and_source_bound(
    tmp_path: Path,
) -> None:
    capability = tmp_path / "capability.pt"
    capability.write_bytes(b"capability")
    output = _write_primitive_unary_artifact(
        tmp_path / "unary.pt",
        scene_id="fern",
        protocol_hash="p" * 64,
        capability_cache=capability,
        capability_source_contract="exact_mpr",
        valid=torch.tensor([True, False, True]),
        primitive_unary_probability=torch.tensor([0.8, 0.0, 0.3]),
        compiler_contract={"prototype_count": 16, "graph_disabled": True},
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["valid_rows"].tolist() == [0, 2]
    assert payload["primitive_unary_probability"].tolist() == [
        torch.tensor(0.8).item(),
        0.0,
        torch.tensor(0.3).item(),
    ]
    assert payload["written_before_target_ground_truth_open"] is True
    assert payload["target_mask_opened"] is False
    assert payload["compiler_contract"]["prototype_count"] == 16


def test_ladder_difference_reports_candidate_minus_reference() -> None:
    summary = _difference(
        torch.tensor([0.0, 0.5, 1.0]),
        torch.tensor([0.0, 0.7, 0.8]),
        threshold=0.6,
    )
    assert abs(summary["mean_absolute_error"] - (0.4 / 3.0)) < 1e-6
    assert abs(summary["threshold_disagreement_fraction"] - (1.0 / 3.0)) < 1e-6
    assert -1.0 <= summary["pearson"] <= 1.0


def test_pre_metric_prediction_receipt_is_complete_and_immutable(
    tmp_path: Path,
) -> None:
    capability = tmp_path / "capability.pt"
    graph = tmp_path / "graph.pt"
    capability.write_bytes(b"capability")
    graph.write_bytes(b"graph")
    scores: dict[str, str] = {}
    score_hashes: dict[str, str] = {}
    for frame_id, value in (("f1", b"one"), ("f2", b"two")):
        score = tmp_path / f"{frame_id}.npy"
        score.write_bytes(value)
        scores[frame_id] = str(score)
        score_hashes[frame_id] = hashlib.sha256(value).hexdigest()
    receipt, digest = _write_pre_metric_prediction_receipt(
        tmp_path / "receipt.json",
        scene_id="fern",
        protocol_hash="p" * 64,
        capability_cache=capability,
        support_graph=graph,
        score_paths=scores,
        score_sha256=score_hashes,
        stage_score_paths={"propagated": scores},
        stage_score_sha256={"propagated": score_hashes},
        method_contract={"graph_policy": "typed"},
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert len(digest) == 64
    assert payload["sealed_before_target_ground_truth_open"] is True
    assert payload["target_mask_opened"] is False
    assert sorted(payload["target_scores"]) == ["f1", "f2"]

    # Exact reuse is accepted; changing even one bound field fails closed.
    _write_pre_metric_prediction_receipt(
        receipt,
        scene_id="fern",
        protocol_hash="p" * 64,
        capability_cache=capability,
        support_graph=graph,
        score_paths=scores,
        score_sha256=score_hashes,
        stage_score_paths={"propagated": scores},
        stage_score_sha256={"propagated": score_hashes},
        method_contract={"graph_policy": "typed"},
    )
    try:
        _write_pre_metric_prediction_receipt(
            receipt,
            scene_id="fern",
            protocol_hash="p" * 64,
            capability_cache=capability,
            support_graph=graph,
            score_paths=scores,
            score_sha256=score_hashes,
            stage_score_paths={"propagated": scores},
            stage_score_sha256={"propagated": score_hashes},
            method_contract={"graph_policy": "legacy"},
        )
    except ValueError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("changed receipt unexpectedly overwrote frozen JSON")


def test_pre_metric_prediction_receipt_binds_explicit_disabled_graph(
    tmp_path: Path,
) -> None:
    capability = tmp_path / "capability.pt"
    capability.write_bytes(b"capability")
    score = tmp_path / "score.npy"
    score.write_bytes(b"sealed")
    score_hash = hashlib.sha256(b"sealed").hexdigest()
    receipt, _ = _write_pre_metric_prediction_receipt(
        tmp_path / "receipt.json",
        scene_id="fern",
        protocol_hash="p" * 64,
        capability_cache=capability,
        support_graph="",
        score_paths={"f1": str(score)},
        score_sha256={"f1": score_hash},
        stage_score_paths={},
        stage_score_sha256={},
        method_contract={"registered_graph_disabled": True},
        graph_disabled=True,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["support_graph"] == {
        "path": None,
        "policy": "disabled_zero_edge_unary_prior_only",
        "sha256": None,
    }
    assert payload["sealed_before_target_ground_truth_open"] is True
