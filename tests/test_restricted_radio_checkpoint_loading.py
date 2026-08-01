from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.models.siglip_projection import (
    OFFICIAL_C_RADIO_V4_H_HALF_SHA256,
)
from radio_gs.utils.immutable_artifacts import (
    load_fixed_radio_checkpoint_payload,
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


OFFICIAL_CHECKPOINT = Path(
    "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"
)
EXPECTED_PROJECTION_DIGEST = (
    "e7caa87ed79d9eca2478c99d5713a800fce6491144235c97874e2d14b810b0c0"
)
EXPECTED_SUMMARY_DIGEST = (
    "00fff7b5cc72f1562c9df70cae2efd4ac9a558d1c449fd904ea3bf64f5986d21"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_mapping_digest(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        tensor = values[key].detach().cpu().contiguous()
        metadata = json.dumps(
            [key, str(tensor.dtype), list(tensor.shape)],
            separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def test_restricted_loader_accepts_namespace_tensor_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "radio_like.pt"
    expected_state = OrderedDict(
        [("weight", torch.tensor([[1.25, -2.5]], dtype=torch.float32))]
    )
    torch.save(
        {
            "state_dict": expected_state,
            "args": argparse.Namespace(arch="radio-test", patch_size=16),
            "arch": "radio-test",
            "version": 1,
        },
        checkpoint,
    )
    expected_sha256 = _sha256(checkpoint)

    payload, observed_sha256, source = load_fixed_radio_checkpoint_payload(
        checkpoint,
        expected_sha256=expected_sha256,
    )

    assert observed_sha256 == expected_sha256
    assert source == checkpoint.resolve()
    assert isinstance(payload["args"], argparse.Namespace)
    assert payload["args"].arch == "radio-test"
    assert list(payload["state_dict"]) == ["weight"]
    assert torch.equal(payload["state_dict"]["weight"], expected_state["weight"])


def test_restricted_loader_checks_sha_before_deserialization(tmp_path: Path) -> None:
    checkpoint = tmp_path / "radio_like.pt"
    torch.save(
        {"state_dict": OrderedDict([("weight", torch.ones(1))])},
        checkpoint,
    )

    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_fixed_radio_checkpoint_payload(
            checkpoint,
            expected_sha256="0" * 64,
        )


def test_restricted_loader_rejects_every_unlisted_global(tmp_path: Path) -> None:
    marker = tmp_path / "must_not_exist"

    class ExecuteIfUnpickled:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    checkpoint = tmp_path / "malicious.pt"
    torch.save({"payload": ExecuteIfUnpickled()}, checkpoint)

    with pytest.raises(Exception, match="forbidden RADIO checkpoint global"):
        load_fixed_radio_checkpoint_payload(
            checkpoint,
            expected_sha256=_sha256(checkpoint),
        )
    assert not marker.exists()


def test_restricted_loader_rejects_final_component_symlink(tmp_path: Path) -> None:
    checkpoint = tmp_path / "radio_like.pt"
    torch.save({"state_dict": OrderedDict()}, checkpoint)
    link = tmp_path / "radio_link.pt"
    link.symlink_to(checkpoint)

    with pytest.raises(ValueError, match="symlink"):
        load_fixed_radio_checkpoint_payload(
            link,
            expected_sha256=_sha256(checkpoint),
        )


def test_sha_bound_project_loader_accepts_legacy_path_and_rejects_code(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy_project.pt"
    torch.save(
        {
            "state_dict": OrderedDict([("weight", torch.ones(1))]),
            "training_config": {"output": tmp_path / "output.pt"},
            "legacy_numpy": {
                "scalar": np.float64(1.25),
                "dtype": np.dtype("float32"),
            },
        },
        checkpoint,
    )
    payload, digest, _ = load_sha_bound_project_checkpoint_mapping(
        checkpoint,
        expected_sha256=_sha256(checkpoint),
    )
    assert digest == _sha256(checkpoint)
    assert payload["training_config"]["output"] == tmp_path / "output.pt"
    assert payload["legacy_numpy"]["scalar"] == np.float64(1.25)
    assert payload["legacy_numpy"]["dtype"] == np.dtype("float32")

    marker = tmp_path / "project_loader_must_not_execute"

    class ExecuteIfUnpickled:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    malicious = tmp_path / "malicious_project.pt"
    torch.save({"payload": ExecuteIfUnpickled()}, malicious)
    with pytest.raises(Exception, match="forbidden project checkpoint global"):
        load_sha_bound_project_checkpoint_mapping(
            malicious,
            expected_sha256=_sha256(malicious),
        )
    assert not marker.exists()


def test_immutable_writers_never_replace_existing_artifacts(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.json"
    write_frozen_json(frozen, {"value": 1})
    write_frozen_json(frozen, {"value": 1})
    with pytest.raises(ValueError, match="existing frozen artifact differs"):
        write_frozen_json(frozen, {"value": 2})
    assert json.loads(frozen.read_text(encoding="utf-8")) == {"value": 1}

    tensor = tmp_path / "tensor.pt"
    write_torch_noclobber(tensor, torch.tensor([1.0]))
    with pytest.raises(FileExistsError, match="already exists"):
        write_torch_noclobber(tensor, torch.tensor([2.0]))
    restored = torch.load(tensor, map_location="cpu", weights_only=True)
    assert torch.equal(restored, torch.tensor([1.0]))


@pytest.mark.skipif(
    not OFFICIAL_CHECKPOINT.is_file(),
    reason="official C-RADIOv4-H checkpoint is not installed",
)
def test_official_projection_and_summary_match_frozen_legacy_values() -> None:
    payload, digest, _ = load_fixed_radio_checkpoint_payload(
        OFFICIAL_CHECKPOINT,
        expected_sha256=OFFICIAL_C_RADIO_V4_H_HALF_SHA256,
    )
    assert digest == OFFICIAL_C_RADIO_V4_H_HALF_SHA256
    state_dict = payload["state_dict"]

    projection: dict[str, torch.Tensor] = {}
    projection_prefix = "_feature_projections.siglip2-g."
    for key, value in state_dict.items():
        if not key.startswith(projection_prefix):
            continue
        output_key = key[len(projection_prefix):]
        if output_key.startswith("mlp.fc1"):
            output_key = output_key.replace("mlp.fc1", "mlp_fc1", 1)
        elif output_key.startswith("mlp.final"):
            output_key = output_key.replace("mlp.final", "mlp_final", 1)
        projection[output_key] = value.float()

    summary_prefix = "_heads.siglip2-g."
    summary = {
        key[len(summary_prefix):]: value.float()
        for key, value in state_dict.items()
        if key.startswith(summary_prefix)
    }

    assert len(projection) == 32
    assert len(summary) == 14
    assert all(value.dtype == torch.float32 for value in projection.values())
    assert all(value.dtype == torch.float32 for value in summary.values())
    assert _tensor_mapping_digest(projection) == EXPECTED_PROJECTION_DIGEST
    assert _tensor_mapping_digest(summary) == EXPECTED_SUMMARY_DIGEST
