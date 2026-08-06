from __future__ import annotations

from pathlib import Path

import pytest
import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.scripts.run_nvos_with_pre_gt_receipt import (
    _atomic_write_no_clobber,
    _build_receipt_payload,
    _canonical_sha256,
    _file_sha256,
    _require_file,
    _wait_for_blocked_target_open,
    _wait_for_child_success,
)


class _Child:
    def __init__(self, *, poll_code=None, wait_code=0):
        self.poll_code = poll_code
        self.wait_code = wait_code

    def poll(self):
        return self.poll_code

    def wait(self, timeout):
        assert timeout >= 1.0
        return self.wait_code


def test_child_exit_before_target_gate_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exited before.*gated: 17"):
        _wait_for_blocked_target_open(_Child(poll_code=17), tmp_path / "missing", 1e30)


@pytest.mark.parametrize("missing", ["primitive", "score"])
def test_missing_prediction_side_artifact_fails(tmp_path: Path, missing: str) -> None:
    primitive = tmp_path / "primitive.pt"
    score = tmp_path / "score.npy"
    torch.save({"primitive_unary_probability": torch.tensor([0.25, 0.75])}, primitive)
    score.write_bytes(b"sealed-score")
    (primitive if missing == "primitive" else score).unlink()
    with pytest.raises(RuntimeError, match="before primitive/score persistence"):
        _build_payload(
            tmp_path,
            primitive=primitive,
            score=score,
            create_predictions=False,
        )


def _build_payload(
    tmp_path: Path,
    *,
    primitive: Path | None = None,
    score: Path | None = None,
    create_predictions: bool = True,
):
    assets = {}
    for name in ("completion", "completion_receipt", "gate", "evaluator", "target"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        assets[name] = path
    primitive = primitive or tmp_path / "primitive.pt"
    score = score or tmp_path / "score.npy"
    if create_predictions and not primitive.exists():
        torch.save(
            {"primitive_unary_probability": torch.tensor([0.25, 0.75])},
            primitive,
        )
    if create_predictions and not score.exists():
        score.write_bytes(b"sealed-score")
    return _build_receipt_payload(
        scene_id="scene",
        command=["python", "evaluator.py", "--frozen"],
        target=assets["target"],
        output_dir=tmp_path,
        primitive=primitive,
        score=score,
        completion=assets["completion"],
        completion_sha256=_file_sha256(assets["completion"]),
        completion_receipt=assets["completion_receipt"],
        completion_receipt_sha256=_file_sha256(assets["completion_receipt"]),
        source_gate=assets["gate"],
        source_gate_sha256=_file_sha256(assets["gate"]),
        evaluator=assets["evaluator"],
        evaluator_sha256=_file_sha256(assets["evaluator"]),
    )


def test_receipt_binds_config_tensor_hash_and_is_no_clobber(tmp_path: Path) -> None:
    payload = _build_payload(tmp_path)
    primitive = torch.load(tmp_path / "primitive.pt", map_location="cpu", weights_only=True)
    assert payload["candidate_config_sha256"] == _canonical_sha256(
        payload["candidate_config"]
    )
    assert payload["primitive_unary"]["score_tensor_sha256"] == tensor_sha256(
        primitive["primitive_unary_probability"]
    )
    assert payload["safety"]["sealed_before_target_ground_truth_open"] is True
    receipt = tmp_path / "receipt.json"
    _atomic_write_no_clobber(receipt, b"first\n")
    _atomic_write_no_clobber(receipt, b"first\n")
    with pytest.raises(ValueError, match="refusing to replace"):
        _atomic_write_no_clobber(receipt, b"different\n")
    assert receipt.read_bytes() == b"first\n"


def test_post_release_child_failure_is_reported() -> None:
    with pytest.raises(RuntimeError, match="failed after GT release: 23"):
        _wait_for_child_success(_Child(wait_code=23), 1e30)


@pytest.mark.parametrize("label", ["completion", "gate", "evaluator"])
def test_upstream_sha_mismatch_fails_closed(tmp_path: Path, label: str) -> None:
    path = tmp_path / label
    path.write_bytes(b"authority")
    with pytest.raises(ValueError, match=f"{label} SHA-256 differs"):
        _require_file(path, "0" * 64, label)
