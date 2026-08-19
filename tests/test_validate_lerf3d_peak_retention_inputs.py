from argparse import Namespace
import hashlib
from pathlib import Path

import torch

from radio_gs.scripts import validate_lerf3d_peak_retention_inputs as validator


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_binds_cache_to_field_renderer_and_authorities(
    tmp_path: Path, monkeypatch
) -> None:
    paths = {}
    for name in (
        "field",
        "renderer",
        "config",
        "method",
        "summary",
        "text",
        "canonical",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path
    monkeypatch.setattr(validator, "METHOD_AUTHORITY_SHA256", _sha(paths["method"]))
    monkeypatch.setattr(validator, "SUMMARY_HEAD_SHA256", _sha(paths["summary"]))
    monkeypatch.setattr(validator, "TEXT_CACHE_SHA256", _sha(paths["text"]))
    monkeypatch.setattr(
        validator, "CANONICAL_CACHE_SHA256", _sha(paths["canonical"])
    )
    metadata = {
        "schema_version": 1,
        "artifact_type": "radio_gs_method_v1_primitive_query_cache",
        "method_id": "radio-gs-method-v1",
        "feature_space": "official_siglip2_summary_descriptor_per_primitive",
        "construction": "canonical_feature_field_decode_then_frozen_official_summary_head_then_l2",
        "query_independent": True,
        "postprocessing": "none",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "text_queries_opened": False,
        "field_checkpoint": {
            "path": str(paths["field"]),
            "sha256": _sha(paths["field"]),
        },
        "renderer_geometry_checkpoint": {
            "path": str(paths["renderer"]),
            "sha256": _sha(paths["renderer"]),
        },
        "method_authority": {
            "path": str(paths["method"]),
            "sha256": _sha(paths["method"]),
        },
        "summary_head": {
            "path": str(paths["summary"]),
            "sha256": _sha(paths["summary"]),
        },
    }
    cache = tmp_path / "cache.pth"
    torch.save(
        {
            "xyz": torch.zeros(2, 3),
            "summary_features": torch.zeros(2, 1536, dtype=torch.float16),
            "valid": torch.ones(2, dtype=torch.bool),
            "metadata": metadata,
        },
        cache,
    )
    args = Namespace(
        scene="figurines",
        primitive_query_cache=str(cache),
        expected_primitive_query_cache_sha256=_sha(cache),
        field=str(paths["field"]),
        expected_field_sha256=_sha(paths["field"]),
        renderer=str(paths["renderer"]),
        expected_renderer_sha256=_sha(paths["renderer"]),
        config=str(paths["config"]),
        expected_config_sha256=_sha(paths["config"]),
        method_authority=str(paths["method"]),
        summary_head=str(paths["summary"]),
        text_cache=str(paths["text"]),
        canonical_cache=str(paths["canonical"]),
    )

    report = validator.validate(args)

    assert report["status"] == "pass"
    assert report["field"]["sha256"] == _sha(paths["field"])
    assert report["primitive_query_cache"]["sha256"] == _sha(cache)
    assert report["benchmark_images_masks_labels_opened_by_preflight"] is False


def test_runner_uses_new_identity_and_explicit_external_cache() -> None:
    runner = Path("radio_gs/scripts/run_lerf3d_peak_retention_guard_full4.sh")
    text = runner.read_text(encoding="utf-8")

    assert "lerf3d_peak_retention_guard_full4_v2_bound" in text
    assert '--external_query_feature_cache "$CACHE"' in text
    assert "validate_lerf3d_peak_retention_inputs.py" in text
    assert "siglip2_lerf_all_exact_official.pt" in text
    assert "siglip2_lerf_all_generic_negatives_exact_official.pt" in text
