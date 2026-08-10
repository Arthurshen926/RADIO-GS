from pathlib import Path

import pytest
import torch

from radio_gs.scripts.materialize_lerf_text_axis_subset import materialize
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    SIGLIP2_MODEL_NAME,
    SIGLIP2_TEXT_CANONICALIZATION,
)
from radio_gs.utils.immutable_artifacts import sha256_file


def _bank(path: Path, queries: list[str]) -> str:
    values = torch.arange(len(queries) * 1536, dtype=torch.float32).reshape(len(queries), 1536)
    torch.save({
        "prompt_templates": ["{query}"],
        "text_encoder": "siglip2",
        "model_name": SIGLIP2_MODEL_NAME,
        "queries": queries,
        "embeddings": values,
    }, path)
    return sha256_file(path)


def test_materialize_copies_source_rows_in_frozen_order(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    order = tmp_path / "order.pt"
    source_sha = _bank(source, ["a", "b", "c"])
    torch.save({"queries": ["c", "a"]}, order)
    output = tmp_path / "subset.pt"
    report = materialize(
        source_bank=source,
        expected_source_bank_sha256=source_sha,
        query_order_cache=order,
        expected_query_order_cache_sha256=sha256_file(order),
        output=output,
    )
    value = torch.load(output, map_location="cpu", weights_only=True)
    original = torch.load(source, map_location="cpu", weights_only=True)
    assert value["queries"] == ["c", "a"]
    assert torch.equal(value["embeddings"], original["embeddings"][[2, 0]])
    assert value["text_canonicalization"] == SIGLIP2_TEXT_CANONICALIZATION
    assert report["source_rows_copied_exactly"] is True


def test_materialize_rejects_missing_query_before_output(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    order = tmp_path / "order.pt"
    source_sha = _bank(source, ["a"])
    torch.save({"queries": ["missing"]}, order)
    output = tmp_path / "subset.pt"
    with pytest.raises(ValueError, match="absent from source"):
        materialize(
            source_bank=source,
            expected_source_bank_sha256=source_sha,
            query_order_cache=order,
            expected_query_order_cache_sha256=sha256_file(order),
            output=output,
        )
    assert not output.exists()
