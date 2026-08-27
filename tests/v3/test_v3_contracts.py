from pathlib import Path

import pytest
import torch

from radio_gs.v3.contracts.method import SUGM_V3_CONTRACT, validate_scene_state
from radio_gs.v3.contracts.static_audit import audit_v3_tree


def test_scene_state_is_exactly_one_d512_plus_five_scalars():
    validate_scene_state(torch.zeros(3, 512), torch.zeros(3, 5), source_authority_sha256="a" * 64)
    with pytest.raises(ValueError, match="D512"):
        validate_scene_state(torch.zeros(3, 528), torch.zeros(3, 5), source_authority_sha256="a" * 64)
    assert SUGM_V3_CONTRACT.gaussian_indexed_high_dimensional_sidecars == 0


def test_v3_core_has_no_historical_import_or_scene_token():
    root = Path(__file__).parents[2] / "radio_gs" / "v3"
    assert audit_v3_tree(root) == []


def test_v3_static_audit_rejects_pre_fusion_local_codes(tmp_path):
    (tmp_path / "bad.py").write_text("value = field.local_codes\n")
    assert audit_v3_tree(tmp_path) == [
        "bad.py: internal pre-fusion local_codes access is forbidden"
    ]
