from __future__ import annotations

import json
from pathlib import Path

import torch

from radio_gs.scripts import build_lerf_per_gaussian_1280d_baseline as baseline


def test_summarize_rows_records_per_gaussian_1280d_protocol() -> None:
    rows = [
        baseline.SceneResult("Figurines", 0.6, 0.4, 10, 120, 100, 3.2),
        baseline.SceneResult("Ramen", 0.8, 0.6, 30, 200, 150, 4.8),
    ]

    summary = baseline.summarize_rows(rows, protocol={"feature_dim": 1280})

    assert summary["protocol"]["feature_dim"] == 1280
    assert summary["macro"]["loc_acc"] == 0.7
    assert summary["macro"]["miou"] == 0.5
    assert summary["weighted"]["loc_acc"] == 0.75
    assert summary["weighted"]["miou"] == 0.55
    assert summary["mean_registered_fraction"] == 0.7917
    assert summary["mean_storage_mib"] == 4.0


def test_markdown_and_latex_name_explicit_1280d_baseline() -> None:
    summary = baseline.summarize_rows(
        [baseline.SceneResult("Figurines", 0.6, 0.4, 10, 120, 100, 3.2)],
        protocol={"feature_dim": 1280},
    )

    markdown = baseline.build_markdown(summary)
    latex = baseline.build_latex_table(summary)

    assert "Per-Gaussian 1280-D explicit RADIO memory" in markdown
    assert "| Figurines | 0.6000 | 0.4000 | 10 | 100/120 | 0.8333 | 3.2 |" in markdown
    assert "\\label{tab:lerf_per_gaussian_1280d_baseline}" in latex


def test_write_outputs_records_all_formats(tmp_path: Path) -> None:
    summary = baseline.summarize_rows(
        [baseline.SceneResult("Figurines", 0.6, 0.4, 10, 120, 100, 3.2)],
        protocol={"feature_dim": 1280},
    )

    paths = baseline.write_outputs(
        summary,
        tmp_path / "report.md",
        tmp_path / "report.json",
        tmp_path / "report.tex",
    )

    assert paths["markdown"].exists()
    assert paths["json"].exists()
    assert paths["latex"].exists()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["rows"][0]["scene"] == "Figurines"


def test_text_embedding_cache_map_parser_uses_scene_specific_paths() -> None:
    parsed = baseline._parse_text_embedding_cache_map(
        "waldo_kitchen=/tmp/waldo.pt,figurines=/tmp/figurines.pt"
    )

    assert parsed == {
        "waldo_kitchen": Path("/tmp/waldo.pt"),
        "figurines": Path("/tmp/figurines.pt"),
    }


def test_resolve_text_embedding_cache_path_prefers_existing_fallback_root(tmp_path: Path) -> None:
    requested_root = tmp_path / "new_text_cache"
    fallback_root = tmp_path / "existing_text_cache"
    fallback_cache = fallback_root / "waldo_kitchen_siglip2_text_embeddings.pt"
    fallback_cache.parent.mkdir(parents=True)
    fallback_cache.write_bytes(b"cache")

    resolved = baseline.resolve_text_embedding_cache_path(
        "waldo_kitchen",
        text_cache_root=requested_root,
        text_embedding_cache_map={},
        fallback_text_cache_roots=(fallback_root,),
        prompt_templates=["{query}"],
    )

    assert resolved == fallback_cache


def test_resolve_text_embedding_cache_path_prefers_explicit_map(tmp_path: Path) -> None:
    mapped = tmp_path / "mapped.pt"

    resolved = baseline.resolve_text_embedding_cache_path(
        "waldo_kitchen",
        text_cache_root=tmp_path / "new_text_cache",
        text_embedding_cache_map={"waldo_kitchen": mapped},
        fallback_text_cache_roots=(tmp_path / "existing_text_cache",),
        prompt_templates=["{query}"],
    )

    assert resolved == mapped


def test_fp16_feature_storage_is_promoted_to_float_for_rendering() -> None:
    stored = torch.ones(2, 1280, dtype=torch.float16)

    rendered = baseline._features_for_render(stored, torch.device("cpu"))

    assert rendered.dtype == torch.float32
    assert rendered.shape == (2, 1280)
