import pytest
import torch
from radio_gs.v4.evaluation.lerf_common import load_text_cache


def test_labeled_canonical_cache_needs_actual_processed_query_receipt(tmp_path):
    path = tmp_path / "text.pt"
    data = {"text_encoder": "siglip2", "text_canonicalization": "official_c_radio_siglip2_g",
            "queries": ["Stainless steel pots", "pour-over vessel"], "embeddings": torch.ones(2, 1536)}
    torch.save(data, path)
    with pytest.raises(ValueError, match="canonical query provenance"):
        load_text_cache(path, data["queries"], torch.device("cpu"))
    data["canonical_queries"] = ["stainless steel pots", "pourover vessel"]
    torch.save(data, path)
    assert load_text_cache(path, data["queries"], torch.device("cpu")).shape == (2, 1536)
    data["embeddings"][0] = 0
    torch.save(data, path)
    with pytest.raises(ValueError, match="zero embeddings"):
        load_text_cache(path, data["queries"], torch.device("cpu"))
